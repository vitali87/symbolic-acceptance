import asyncio
import json
import unittest

from helpers import DIFF, FakeHTTP, json_response, message

import symbolic_acceptance as sa
from symbolic_acceptance import http
from symbolic_acceptance.claude import Claude, ClaudeError
from symbolic_acceptance.github import GitHub
from symbolic_acceptance.judge import Judge, Verdict


def verdict(answer, quotes=(), missing="", tampering=False, reasoning="looked"):
    return message(
        {
            "reasoning": reasoning,
            "answer": answer,
            "quotes": list(quotes),
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
                        "quotes": {"type": "array", "items": {"type": "string"}},
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
        self.assertIn("only lines the change leaves as they were", answer.reasoning)

    def test_absence_needs_no_quote_for_yes(self):
        j, fake = judge(verdict("YES", [], reasoning="no public signature changed"))
        absent = sa.Question(
            "api_stable", "Is the API unchanged?", "llm", options={"kind": "absence"}
        )
        self.assertTrue(ask(j, question=absent).is_met)
        asked = fake.body()["messages"][0]["content"][1]["text"]
        self.assertIn("This question is about absence.", asked)

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

    def test_truncated_reply_is_an_error(self):
        for stop in ("max_tokens", "model_context_window_exceeded"):
            j, _ = judge(message({}, stop=stop))
            with self.assertRaisesRegex(ClaudeError, f"stopped early \\({stop}"):
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
        fake = FakeHTTP(json_response({"error": {"message": "bad schema"}}, 400))
        with self.assertRaisesRegex(http.HTTPError, "HTTP 400: bad schema"):
            http.request(
                "POST",
                "https://example.test/x",
                transport=fake,
                json_body={},
                sleep=lambda s: None,
            )
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

        with self.assertRaisesRegex(http.HTTPError, "connection failed"):
            http.request("POST", "https://example.test/x", transport=timeout)

    def test_retry_after_may_be_a_date(self):
        slept = []
        past = "Wed, 21 Oct 2015 07:28:00 GMT"
        fake = FakeHTTP(
            json_response({}, 503, {"retry-after": past}), json_response({})
        )
        http.request(
            "GET", "https://example.test/x", transport=fake, sleep=slept.append
        )
        self.assertEqual(slept, [0.0], "a date already past means now")

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


if __name__ == "__main__":
    unittest.main()
