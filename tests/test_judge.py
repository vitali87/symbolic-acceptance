import asyncio
import json
import unittest

from helpers import FakeHTTP, json_response
from test_diffs_checks import DIFF

import symbolic_acceptance as sa
from symbolic_acceptance import http
from symbolic_acceptance.claude import Claude, ClaudeError
from symbolic_acceptance.judge import Judge


def message(data, stop="end_turn", model="claude-opus-5-5"):
    return json_response(
        {
            "model": model,
            "stop_reason": stop,
            "stop_details": {"category": "cyber"} if stop == "refusal" else None,
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": json.dumps(data)},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )


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


def ask(judge_, diff=DIFF, issue="Logins are not throttled"):
    checklist = sa.loads(
        '[q.locks_account]\nask = "Does it lock accounts?"\nby = "llm"\n'
        '[[require]]\nthat = "locks_account"\nfail = "x"\n'
    )
    context = sa.Context(
        checklist, sa.Attempt("abc123", 1, diff=diff, claims="trust me"), {}, issue
    )
    return asyncio.run(judge_([QUESTION], context))["locks_account"]


class RequestTests(unittest.TestCase):
    def test_request_shape(self):
        j, fake = judge(verdict("YES", ["lock(user)  "]))
        ask(j)
        body = fake.body()
        self.assertEqual(body["model"], "claude-opus-5-5")
        self.assertEqual(body["fallbacks"], "default")
        self.assertEqual(body["output_config"]["effort"], "high")
        self.assertEqual(body["output_config"]["format"]["type"], "json_schema")
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
        other, _ = judge()
        self.assertNotEqual(j._boundary("abc123", ()), other._boundary("abc123", ()))

    def test_oauth_token(self):
        fake = FakeHTTP(verdict("NO", missing="x"))
        claude = Claude(auth_token="tok", transport=fake)
        ask(Judge(claude))
        self.assertEqual(fake.header("authorization"), "Bearer tok")
        self.assertIn("oauth-2025-04-20", fake.header("anthropic-beta"))


class VerdictTests(unittest.TestCase):
    def test_supported_yes_carries_located_evidence(self):
        j, _ = judge(verdict("YES", ["if failures(user) > MAX_FAILURES:"]))
        answer = ask(j)
        self.assertTrue(answer.is_met)
        self.assertEqual(
            answer.evidence, ("src/auth.py:12: if failures(user) > MAX_FAILURES:",)
        )
        self.assertEqual(answer.by, "llm:claude-opus-5-5")

    def test_unsupported_yes_is_retried_then_counts_as_no(self):
        j, fake = judge(
            verdict("YES", ["rate_limit(user, window=60)"]), verdict("YES", [])
        )
        answer = ask(j)
        self.assertFalse(answer.is_met)
        self.assertIn("quoted nothing", answer.reasoning)
        self.assertEqual(len(fake.requests), 2)

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
        j, _ = judge(message({}, stop="max_tokens"))
        with self.assertRaises(ClaudeError):
            ask(j)

    def test_samples_must_agree_on_yes(self):
        j, fake = judge(
            verdict("YES", ["lock(user)"]), verdict("NO", missing="unsure"), samples=3
        )
        answer = ask(j)
        self.assertFalse(answer.is_met)
        self.assertIn("[1/2 samples YES]", answer.reasoning)
        self.assertEqual(len(fake.requests), 2, "one NO decides; no third sample")

    def test_oversized_diff_goes_to_a_person(self):
        j, fake = judge(max_diff_chars=100)
        with self.assertRaisesRegex(sa.Escalate, "a person must review"):
            ask(j)
        self.assertEqual(fake.requests, [])

    def test_no_credentials(self):
        import os

        saved = {
            k: os.environ.pop(k)
            for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
            if k in os.environ
        }
        try:
            with self.assertRaisesRegex(ClaudeError, "no credentials"):
                Claude(transport=FakeHTTP()).json(system="s", content="c", schema={})
        finally:
            os.environ.update(saved)


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
        self.assertEqual(response.json(), {"ok": True})
        self.assertEqual(slept, [3.0, 2.0])

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
