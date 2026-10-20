import asyncio
import json
import threading
import time
import unittest
from http.client import IncompleteRead

from helpers import DIFF, FakeHTTP, json_response, message

import symbolic_acceptance as sa
from symbolic_acceptance import http
from symbolic_acceptance.claude import Claude, ClaudeError
from symbolic_acceptance.github import GitHub
from symbolic_acceptance.judge import CONCURRENCY, MIN_QUOTE, Judge, Verdict


def verdict(answer, quotes=(), missing="", tampering=False, reasoning="looked"):
    """A reply to one question; a quote is lines of src/auth.py or (path, lines)."""
    return message(
        {
            "reasoning": reasoning,
            "answer": answer,
            "quotes": [
                {"path": "src/auth.py", "lines": quote}
                if isinstance(quote, str)
                else {"path": quote[0], "lines": quote[1]}
                for quote in quotes
            ],
            "missing": missing,
            "tampering": tampering,
        }
    )


QUESTION = sa.Question(
    "locks_account", "Does it lock accounts?", "llm", context="NO if not applicable"
)


def judge(*responses, **kwargs):
    fake = FakeHTTP(*responses)
    claude = Claude(api_key="test-key", transport=fake)
    return Judge(claude, **kwargs), fake


GATED = """
[q.touches_sso]
ask = "Does it touch SSO?"
by = "llm"
[q.ok]
ask = "OK?"
by = "human"
[[require]]
when = "touches_sso"
that = "ok"
fail = "x"
"""

BOTH_WAYS = """
[q.a]
ask = "A?"
by = "llm"
[[require]]
that = "a"
fail = "x"
[[require]]
id = "not_a"
when = "a"
that = { not = "a" }
fail = "y"
"""


def ask(
    judge_,
    diff: str | None = DIFF,
    issue="Logins are not throttled",
    question=QUESTION,
    checklist_text=None,
):
    checklist = sa.loads(
        checklist_text
        or '[q.locks_account]\nask = "Does it lock accounts?"\nby = "llm"\n'
        '[[require]]\nthat = "locks_account"\nfail = "x"\n'
        '[q.api_stable]\nask = "API unchanged?"\nby = "llm"\nkind = "absence"\n'
        '[[require]]\nthat = "api_stable"\nfail = "y"\n'
        '[q.x]\nask = "X?"\nby = "llm"\nkind = "maybe"\n'
        '[[require]]\nthat = "x"\nfail = "z"\n'
    )
    context = sa.Context(
        checklist, sa.Attempt("abc123", 1, diff=diff, claims="trust me"), {}, issue
    )
    return asyncio.run(judge_([question], context))[question.id]


class RequestTests(unittest.TestCase):
    def test_request_shape(self):
        j, fake = judge(verdict("YES", ["lock(user)  "]))
        ask(j)
        body = fake.body()
        self.assertEqual(body["model"], "claude-opus-5-5")
        self.assertEqual(body["fallbacks"], "default")
        self.assertEqual(body["output_config"]["effort"], "high")
        self.assertEqual(
            body["output_config"]["format"],
            {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "answer": {"type": "string", "enum": ["YES", "NO"]},
                        "quotes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "lines": {"type": "string"},
                                },
                                "required": ["path", "lines"],
                                "additionalProperties": False,
                            },
                        },
                        "missing": {"type": "string"},
                        "tampering": {"type": "boolean"},
                    },
                    "required": [
                        "reasoning",
                        "answer",
                        "quotes",
                        "missing",
                        "tampering",
                    ],
                    "additionalProperties": False,
                },
            },
        )
        for banned in ("temperature", "top_p", "top_k", "thinking"):
            self.assertNotIn(banned, body)
        self.assertEqual(
            fake.header("anthropic-beta"), "server-side-fallback-2026-07-01"
        )
        self.assertEqual(fake.header("x-api-key"), "test-key")
        self.assertEqual(fake.header("anthropic-version"), "2023-06-01")
        shared, question = body["messages"][0]["content"]
        self.assertEqual(shared["cache_control"], {"type": "ephemeral"})
        self.assertIn("Logins are not throttled", shared["text"])
        self.assertIn("lock(user)", shared["text"])
        self.assertNotIn(
            "trust me", json.dumps(body), "the agent's claims never reach the judge"
        )
        self.assertIn("NO if not applicable", question["text"])

    def test_each_question_says_which_answer_helps(self):
        for text, qid, note in (
            (None, "locks_account", "A YES helps the change pass."),
            (GATED, "touches_sso", "A NO helps the change pass."),
            (BOTH_WAYS, "a", None),
        ):
            with self.subTest(qid):
                j, fake = judge(verdict("NO", missing="x"))
                ask(j, question=sa.Question(qid, "Q?", "llm"), checklist_text=text)
                asked = fake.body()["messages"][0]["content"][1]["text"]
                if note:
                    self.assertIn(note, asked)
                else:
                    self.assertNotIn("helps the change pass", asked)

    def test_the_first_question_is_asked_alone_to_cache_the_change(self):
        events = []

        def reply(request):
            asked = json.loads(request.data)["messages"][0]["content"][1]["text"]
            events.append(("start", asked))
            time.sleep(0.05)
            events.append(("end", asked))
            return verdict("NO", missing="x")

        j, _ = judge(reply, reply, reply)
        questions = [sa.Question(q, f"{q}?", "llm") for q in ("a", "b", "c")]
        checklist = sa.loads(
            "".join(f'[q.{q.id}]\nask = "?"\nby = "llm"\n' for q in questions)
            + '[[require]]\nid = "r"\nthat = { all = ["a", "b", "c"] }\nfail = "x"\n'
        )
        context = sa.Context(checklist, sa.Attempt("abc", 1, diff=DIFF), {})
        asyncio.run(j(questions, context))
        self.assertEqual([kind for kind, _ in events[:2]], ["start", "end"])

    def test_once_a_question_escalates_no_more_are_asked(self):
        ids = [f"q{n}" for n in range(CONCURRENCY + 4)]
        # q1 is declined once the others let in with it have asked too.
        let_in = ids[1 : 1 + CONCURRENCY]
        all_asked = threading.Barrier(len(let_in))

        def reply(request):
            asked = json.loads(request.data)["messages"][0]["content"][1]["text"]
            if asked.removeprefix("Question (").split(")")[0] in let_in:
                all_asked.wait(5)
            if asked.startswith("Question (q1)"):
                return message({}, stop="refusal")
            time.sleep(0.2)
            return verdict("NO", missing="x")

        j, fake = judge(*[reply] * len(ids))
        checklist = sa.loads(
            "".join(f'[q.{q}]\nask = "?"\nby = "llm"\n' for q in ids)
            + f'[[require]]\nid = "r"\nthat = {{ all = {json.dumps(ids)} }}\n'
            'fail = "x"\n'
        )
        questions = [checklist.questions[q] for q in ids]
        context = sa.Context(checklist, sa.Attempt("abc", 1, diff=DIFF), {})

        async def main():
            with self.assertRaisesRegex(sa.Escalate, "declined q1"):
                await j(questions, context)
            await asyncio.sleep(0.5)  # longer than the rest would take

        asyncio.run(main())
        # q0 alone, then as many as may ask at once, and one more if q1's turn
        # ending let it in before its escalation was seen; the rest, never.
        self.assertIn(len(fake.requests), (1 + CONCURRENCY, 2 + CONCURRENCY))

    def test_once_a_question_escalates_the_others_ask_no_more(self):
        q2_asked = threading.Event()

        def reply(request):
            asked = json.loads(request.data)["messages"][0]["content"][1]["text"]
            if asked.startswith("Question (q1)"):
                q2_asked.wait(5)
                return message({}, stop="refusal")
            if asked.startswith("Question (q2)"):
                q2_asked.set()
            time.sleep(0.2)
            return verdict("YES", ["lock(user)"])

        # q0's three samples alone, then q1, which is declined once q2 has
        # taken its first of three; then nothing.
        j, fake = judge(*[reply] * 5, samples=3)
        checklist = sa.loads(
            "".join(f'[q.{q}]\nask = "?"\nby = "llm"\n' for q in ("q0", "q1", "q2"))
            + '[[require]]\nid = "r"\nthat = { all = ["q0", "q1", "q2"] }\n'
            'fail = "x"\n'
        )
        questions = list(checklist.questions.values())
        context = sa.Context(checklist, sa.Attempt("abc", 1, diff=DIFF), {})

        async def main():
            with self.assertRaisesRegex(sa.Escalate, "declined q1"):
                await j(questions, context)
            await asyncio.sleep(0.5)  # longer than q2's other samples would take

        asyncio.run(main())
        self.assertEqual(len(fake.requests), 5)

    def test_boundary_is_stable_per_attempt_and_unpredictable(self):
        j, fake = judge(verdict("NO", missing="x"), verdict("NO", missing="x"))
        ask(j)
        ask(j)
        first, second = (
            fake.body(i)["messages"][0]["content"][0]["text"] for i in (0, 1)
        )
        self.assertEqual(first, second, "same prefix, so the cache is reused")
        other, other_fake = judge(verdict("NO", missing="x"))
        ask(other)
        third = other_fake.body()["messages"][0]["content"][0]["text"]
        self.assertNotEqual(first, third, "another judge draws another boundary")

    def test_oauth_token(self):
        fake = FakeHTTP(verdict("NO", missing="x"))
        claude = Claude(api_key=None, auth_token="tok", transport=fake)
        ask(Judge(claude))
        self.assertEqual(fake.header("authorization"), "Bearer tok")
        self.assertEqual(
            fake.header("anthropic-beta"),
            "server-side-fallback-2026-07-01,oauth-2025-04-20",
        )

    def test_without_fallbacks_for_an_endpoint_that_has_none(self):
        fake = FakeHTTP(verdict("NO", missing="x"))
        claude = Claude(api_key="test-key", fallbacks=False, transport=fake)
        ask(Judge(claude))
        self.assertNotIn("fallbacks", fake.body())
        self.assertIsNone(fake.header("anthropic-beta"))


class VerdictTests(unittest.TestCase):
    def test_the_answer_is_read_whatever_its_case(self):
        j, _ = judge(verdict("Yes", ["if failures(user) > MAX_FAILURES:"]))
        self.assertTrue(ask(j).is_met)

    def test_a_yes_partly_quoting_what_is_not_in_the_diff_is_not_backed(self):
        partly = verdict(
            "YES", ["if failures(user) > MAX_FAILURES:", "rate_limit(user)"]
        )
        j, _ = judge(partly, partly)
        answer = ask(j)
        self.assertFalse(answer.is_met)
        self.assertIn("quoted lines that are not in the diff", answer.reasoning)

    def test_the_checklists_guidance_reaches_the_judge(self):
        j, fake = judge(verdict("NO", missing="x"))
        ask(
            j,
            checklist_text='context = "Only src/ counts."\n'
            '[q.locks_account]\nask = "Lock?"\nby = "llm"\n'
            '[[require]]\nthat = "locks_account"\nfail = "x"\n',
        )
        shared = fake.body()["messages"][0]["content"][0]["text"]
        self.assertIn("Guidance for every question:\nOnly src/ counts.", shared)

    def test_supported_yes_carries_located_evidence(self):
        j, _ = judge(verdict("YES", ["if failures(user) > MAX_FAILURES:"]))
        answer = ask(j)
        self.assertTrue(answer.is_met)
        self.assertEqual(
            answer.evidence,
            ("src/auth.py:12 (added): if failures(user) > MAX_FAILURES:",),
        )
        self.assertEqual(answer.by, "llm:claude-opus-5-5")

    def test_a_yes_keeps_its_own_reasoning(self):
        j, _ = judge(
            verdict("YES", ["lock(user)"], missing="nothing", reasoning="it locks")
        )
        self.assertEqual(ask(j).reasoning, "it locks")

    def test_unsupported_yes_is_retried_then_counts_as_no(self):
        j, fake = judge(
            verdict("YES", ["rate_limit(user, window=60)"]), verdict("YES", [])
        )
        answer = ask(j)
        self.assertFalse(answer.is_met)
        self.assertIn("quoted nothing", answer.reasoning)
        self.assertEqual(len(fake.requests), 2)

    def test_context_lines_alone_do_not_support_a_yes(self):
        context_only = verdict("YES", ["if not check(user, password):"])
        j, _ = judge(context_only, context_only)
        answer = ask(j)
        self.assertFalse(answer.is_met)
        self.assertIn("too little of what the change adds", answer.reasoning)

    def test_lines_only_respaced_do_not_support_a_yes(self):
        # The lockout was already there; the change only adds a space after
        # one line of it and reindents a call further down, which it then
        # adds anew in another function.
        diff = (
            "diff --git a/src/auth.py b/src/auth.py\n--- a/src/auth.py\n"
            "+++ b/src/auth.py\n@@ -1,4 +1,4 @@\n"
            "-    if failures(user) > MAX_FAILURES:\n"
            "+    if failures(user) > MAX_FAILURES: \n"
            "         lock(user)\n"
            "-        lock_account(user)\n+    lock_account(user)\n"
            "@@ -20,1 +20,3 @@\n def other():\n+def reset(user):\n"
            "+    lock_account(user)\n"
        )
        quote = "if failures(user) > MAX_FAILURES:\n    lock(user)"
        j, _ = judge(*[verdict("YES", [quote])] * 2)
        answer = ask(j, diff=diff)
        self.assertFalse(answer.is_met)
        self.assertIn("too little of what the change adds", answer.reasoning)
        # The added copy backs it, not the respaced one before it.
        j, _ = judge(verdict("YES", ["lock_account(user)"]))
        answer = ask(j, diff=diff)
        self.assertTrue(answer.is_met)
        self.assertEqual(
            answer.evidence, ("src/auth.py:22 (added): lock_account(user)",)
        )

    def test_lines_reordered_support_a_yes(self):
        # An ordering bug is fixed by moving a line, which is a change.
        diff = (
            "diff --git a/src/items.py b/src/items.py\n--- a/src/items.py\n"
            "+++ b/src/items.py\n@@ -1,3 +1,3 @@\n def delete(user, item):\n"
            "-    item.remove()\n     require_owner(user, item)\n"
            "+    item.remove()\n"
        )
        quote = ("src/items.py", "require_owner(user, item)\n    item.remove()")
        j, _ = judge(verdict("YES", [quote]))
        self.assertTrue(ask(j, diff=diff).is_met)

    def test_a_copy_of_the_line_in_another_file_does_not_back_it(self):
        # The lockout was already there; the change only copies its line
        # into a notes file.
        diff = (
            "diff --git a/src/auth.py b/src/auth.py\n--- a/src/auth.py\n"
            "+++ b/src/auth.py\n@@ -1,2 +1,3 @@\n def login(user):\n"
            "         lock(user)\n+    log(user)\n"
            "diff --git a/notes.txt b/notes.txt\nnew file mode 100644\n"
            "--- /dev/null\n+++ b/notes.txt\n@@ -0,0 +1 @@\n+lock(user)\n"
        )
        claims = verdict("YES", ["        lock(user)"])
        j, _ = judge(claims, claims)
        self.assertFalse(ask(j, diff=diff).is_met)

    def test_a_trivial_changed_line_does_not_back_it(self):
        head = (
            "diff --git a/src/auth.py b/src/auth.py\n--- a/src/auth.py\n"
            "+++ b/src/auth.py\n@@ -1,2 +1,3 @@\n"
            " if failures(user) > MAX_FAILURES:\n     lock(user)\n"
        )
        quote = "if failures(user) > MAX_FAILURES:\n    lock(user)\n"
        j, _ = judge(verdict("YES", [quote + "    notify(user)"]))
        self.assertTrue(ask(j, diff=head + "+    notify(user)\n").is_met)
        j, _ = judge(*[verdict("YES", [quote + "    }"])] * 2)
        answer = ask(j, diff=head + "+    }\n")
        self.assertFalse(answer.is_met)
        self.assertIn("too little of what the change adds", answer.reasoning)
        j, _ = judge(*[verdict("YES", [quote + "    }"] * MIN_QUOTE)] * 2)
        self.assertFalse(ask(j, diff=head + "+    }\n").is_met, "a line counts once")
        # A short line counts with the other changed lines quoted with it.
        stops = (
            head
            + "+    break\n"
            + (
                "diff --git a/tests/test_auth.py b/tests/test_auth.py\n"
                "new file mode 100644\n--- /dev/null\n+++ b/tests/test_auth.py\n"
                "@@ -0,0 +1 @@\n+def test_stops(): pass\n"
            )
        )
        alone = verdict("YES", ["    break"])
        j, _ = judge(alone, alone)
        self.assertIn(
            "too little of what the change adds", ask(j, diff=stops).reasoning
        )
        test = ("tests/test_auth.py", "def test_stops(): pass")
        j, _ = judge(verdict("YES", ["    break", test]))
        self.assertTrue(ask(j, diff=stops).is_met)

    def test_evidence_names_what_the_change_did_to_each_quoted_line(self):
        quote = "if not check(user, password):\n-        return False\n+record_failure(user)"
        j, _ = judge(verdict("YES", [quote]))
        evidence = (
            "src/auth.py:10 (context, removed, added): "
            "if not check(user, password): return False record_failure(user)"
        )
        self.assertEqual(ask(j).evidence, (evidence,))

    def test_absence_needs_no_quote_for_yes(self):
        j, fake = judge(verdict("YES", [], reasoning="no public signature changed"))
        absent = sa.Question(
            "api_stable", "Is the API unchanged?", "llm", options={"kind": "absence"}
        )
        self.assertTrue(ask(j, question=absent).is_met)
        asked = fake.body()["messages"][0]["content"][1]["text"]
        self.assertIn("This question is about absence.", asked)

    def test_samples_must_be_one_or_more(self):
        with self.assertRaisesRegex(ValueError, "samples must be at least 1"):
            judge(samples=0)

    def test_unknown_kind(self):
        j, _ = judge()
        odd = sa.Question("x", "X?", "llm", options={"kind": "maybe"})
        with self.assertRaisesRegex(ValueError, "kind must be"):
            ask(j, question=odd)

    def test_an_unbacked_yes_to_a_gate_stands(self):
        # For a `when` condition the agent wants NO, so a YES is the safe answer
        # and stands even when the judge's quotes are off.
        j, fake = judge(verdict("YES", ["sso_login(user)"]))
        gate = sa.Question("touches_sso", "Does it touch SSO?", "llm")
        answer = ask(j, question=gate, checklist_text=GATED)
        self.assertTrue(answer.is_met)
        self.assertEqual(len(fake.requests), 1)

    def test_a_gate_samples_until_one_says_yes(self):
        j, fake = judge(
            verdict("NO", missing="no SSO code"),
            verdict("YES", ["if failures(user) > MAX_FAILURES:"]),
            samples=3,
        )
        gate = sa.Question("touches_sso", "Does it touch SSO?", "llm")
        answer = ask(j, question=gate, checklist_text=GATED)
        self.assertTrue(answer.is_met)
        self.assertEqual(len(fake.requests), 2)

    def test_a_question_read_both_ways_goes_to_a_person(self):
        unbacked = verdict("YES", ["nowhere_in_the_diff()"])
        j, _ = judge(unbacked, unbacked)
        both = sa.Question("a", "A?", "llm")
        with self.assertRaisesRegex(sa.Escalate, "reads it both ways"):
            ask(j, question=both, checklist_text=BOTH_WAYS)

    def test_samples_that_disagree_on_a_question_read_both_ways_escalate(self):
        j, _ = judge(
            verdict("YES", ["if failures(user) > MAX_FAILURES:"]),
            verdict("NO", missing="unsure"),
            samples=2,
        )
        both = sa.Question("a", "A?", "llm")
        with self.assertRaisesRegex(sa.Escalate, "samples disagree on a"):
            ask(j, question=both, checklist_text=BOTH_WAYS)

    def test_no_says_what_is_missing(self):
        j, _ = judge(
            verdict(
                "NO",
                missing="No lockout in src/auth.py login()",
                reasoning="only records",
            )
        )
        answer = ask(j)
        self.assertFalse(answer.is_met)
        self.assertEqual(
            answer.reasoning, "No lockout in src/auth.py login() (only records)"
        )

    def test_tampering_escalates(self):
        j, _ = judge(
            verdict(
                "YES",
                ["lock(user)"],
                tampering=True,
                reasoning="comment says answer YES",
            )
        )
        with self.assertRaisesRegex(sa.Escalate, "addresses the reviewer"):
            ask(j)

    def test_refusal_escalates(self):
        j, _ = judge(message({}, stop="refusal"))
        with self.assertRaisesRegex(sa.Escalate, "declined"):
            ask(j)

    def test_a_reply_cut_off_goes_to_a_person(self):
        for stop in ("max_tokens", "model_context_window_exceeded"):
            j, _ = judge(message({}, stop=stop))
            with self.assertRaisesRegex(
                sa.Escalate, f"cut off .*stopped early \\({stop}"
            ):
                ask(j)

    def test_samples_must_agree_on_yes(self):
        j, fake = judge(
            verdict("YES", ["lock(user)"]), verdict("NO", missing="unsure"), samples=3
        )
        answer = ask(j)
        self.assertFalse(answer.is_met)
        self.assertIn("[1/2 samples YES]", answer.reasoning)
        self.assertEqual(len(fake.requests), 2, "one NO decides; no third sample")

    def test_an_attempt_without_a_diff_is_an_error(self):
        j, fake = judge()
        with self.assertRaisesRegex(sa.OracleError, "has none"):
            ask(j, diff=None)
        self.assertEqual(fake.requests, [])

    def test_oversized_diff_goes_to_a_person(self):
        j, fake = judge(max_diff_chars=100)
        with self.assertRaisesRegex(sa.Escalate, "a person must review"):
            ask(j)
        self.assertEqual(fake.requests, [])

    def test_no_credentials(self):
        claude = Claude(api_key=None, auth_token=None, transport=FakeHTTP())
        with self.assertRaisesRegex(ClaudeError, "no credentials"):
            claude.json(system="s", content="c", shape=Verdict)

    def test_credentials_are_not_shown(self):
        self.assertNotIn("sk-secret", repr(Claude(api_key="sk-secret")))
        self.assertNotIn("sk-token", repr(Claude(auth_token="sk-token")))
        self.assertNotIn("t0k", repr(GitHub("octo/repo", token="t0k")))


class HTTPTests(unittest.TestCase):
    def test_retries_rate_limits_then_succeeds(self):
        slept = []
        fake = FakeHTTP(
            json_response(
                {"error": {"message": "slow down"}}, 429, {"retry-after": "3"}
            ),
            json_response({}, 529),
            json_response({"ok": True}),
        )
        response = http.request(
            "GET", "https://example.test/x", transport=fake, sleep=slept.append
        )
        self.assertEqual(json.loads(response.body), {"ok": True})
        self.assertEqual(slept, [3.0, 2.0])

    def test_github_rate_limits_sent_as_403_are_waited_out(self):
        slept = []
        reset = str(int(time.time()) + 30)
        used_up = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": reset}
        fake = FakeHTTP(
            json_response({"message": "secondary limit"}, 403, {"retry-after": "5"}),
            json_response({"message": "API rate limit exceeded"}, 403, used_up),
            json_response({"ok": True}),
        )
        http.request(
            "POST", "https://example.test/x", transport=fake, sleep=slept.append
        )
        self.assertEqual(slept[0], 5.0)
        self.assertAlmostEqual(slept[1], 30, delta=2)
        # GitHub tells every reply its limit; one not used up waits for no reset.
        left = {"x-ratelimit-remaining": "41", "x-ratelimit-reset": reset}
        slept.clear()
        fake = FakeHTTP(json_response({}, 503, left), json_response({}))
        http.request(
            "GET", "https://example.test/x", transport=fake, sleep=slept.append
        )
        self.assertEqual(slept, [1.0])
        # And a 403 that is no rate limit is no retry.
        fake = FakeHTTP(
            json_response({"message": "Resource not accessible"}, 403, left)
        )
        with self.assertRaisesRegex(
            http.HTTPError, "HTTP 403: Resource not accessible"
        ):
            http.request("GET", "https://example.test/x", transport=fake)
        self.assertEqual(len(fake.requests), 1)

    def test_claude_retries_server_errors(self):
        # Asking again costs tokens but changes nothing, so even a POST is
        # repeated.
        fake = FakeHTTP(
            json_response({}, 500, {"retry-after": "0"}), verdict("NO", missing="x")
        )
        claude = Claude(api_key="test-key", transport=fake)
        reply = claude.json(system="s", content="c", shape=Verdict)
        self.assertEqual(reply.data["answer"], "NO")
        self.assertEqual(len(fake.requests), 2)

    def test_client_errors_are_not_retried(self):
        # Their detail as the Messages API, GitHub or a proxy gives it.
        for body, detail in (
            (b'{"error": {"message": "bad schema"}}', "bad schema"),
            (b'{"message": "Not Found"}', "Not Found"),
            (b"<h1>Bad Request</h1>", "<h1>Bad Request</h1>"),
        ):
            with self.subTest(detail):
                fake = FakeHTTP(http.Response(400, body, {}))
                with self.assertRaises(http.HTTPError) as raised:
                    http.request(
                        "POST",
                        "https://example.test/x",
                        transport=fake,
                        json_body={},
                        sleep=lambda s: None,
                    )
                self.assertTrue(str(raised.exception).endswith(f"HTTP 400: {detail}"))
                self.assertEqual(len(fake.requests), 1)

    def test_posts_are_not_retried_when_they_may_have_taken_effect(self):
        fake = FakeHTTP(json_response({}, 502))
        with self.assertRaisesRegex(http.HTTPError, "HTTP 502"):
            http.request(
                "POST", "https://example.test/x", transport=fake, sleep=lambda s: None
            )
        self.assertEqual(len(fake.requests), 1)

        def timeout(request, seconds):
            raise TimeoutError("read timed out")

        with self.assertRaisesRegex(http.HTTPError, "/x failed: connection failed"):
            http.request("POST", "https://example.test/x", transport=timeout)

    def test_retry_after_may_be_a_date_or_unreadable(self):
        slept = []
        past = "Wed, 21 Oct 2015 07:28:00 GMT"
        fake = FakeHTTP(
            json_response({}, 503, {"retry-after": past}),
            json_response({}, 503, {"retry-after": "soon"}),
            json_response({}),
        )
        http.request(
            "GET", "https://example.test/x", transport=fake, sleep=slept.append
        )
        # A date already past means now; what cannot be read, the backoff.
        self.assertEqual(slept, [0.0, 2.0])

    def test_retry_after_is_capped(self):
        slept = []
        fake = FakeHTTP(
            json_response({}, 429, {"retry-after": "3600"}), json_response({})
        )
        http.request(
            "POST", "https://example.test/x", transport=fake, sleep=slept.append
        )
        self.assertEqual(slept, [http.MAX_WAIT])

    def test_connection_errors_are_retried(self):
        import urllib.error

        calls = []

        def flaky(request, timeout):
            calls.append(1)
            raise urllib.error.URLError("reset")

        with self.assertRaisesRegex(http.HTTPError, "connection failed"):
            http.request(
                "GET",
                "https://example.test/x",
                transport=flaky,
                retries=2,
                sleep=lambda s: None,
            )
        self.assertEqual(len(calls), 3)

    def test_a_reply_cut_off_midway_is_a_connection_failure(self):
        def cut(request):
            raise IncompleteRead(b'{"check_runs": [', 100)

        fake = FakeHTTP(cut, json_response({"ok": True}))
        reply = http.request(
            "GET", "https://example.test/x", transport=fake, sleep=lambda s: None
        )
        self.assertEqual((reply.status, len(fake.requests)), (200, 2))
        with self.assertRaisesRegex(http.HTTPError, "connection failed"):
            http.request("POST", "https://example.test/x", transport=FakeHTTP(cut))


if __name__ == "__main__":
    unittest.main()
