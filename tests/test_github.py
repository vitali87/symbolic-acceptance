import asyncio
import unittest

from helpers import FakeHTTP, json_response

import symbolic_acceptance as sa
from symbolic_acceptance.github import (
    REPORT_MARKER,
    GitHub,
    GitHubChecks,
    Reviews,
    post_report,
)


def gh(*responses):
    fake = FakeHTTP(*responses)
    return GitHub("octo/repo", token="t0k", transport=fake), fake


def context(ref="abc123"):
    checklist = sa.loads(
        '[q.x]\nask = "X?"\nby = "ci"\n[[require]]\nthat = "x"\nfail = "x"\n'
    )
    return sa.Context(checklist, sa.Attempt(ref, 1), {})


def run(
    name,
    status="completed",
    conclusion: str | None = "success",
    run_id=1,
    app="github-actions",
):
    return {
        "id": run_id,
        "name": name,
        "app": {"slug": app},
        "status": status,
        "conclusion": conclusion,
        "html_url": f"https://github.com/octo/repo/runs/{run_id}",
        "output": {"title": f"{name} {conclusion}", "summary": None},
    }


def comment(
    cid,
    login,
    body,
    *,
    association="MEMBER",
    user_type="User",
    at="2026-10-18T10:00:00Z",
):
    return {
        "id": cid,
        "user": {"login": login, "type": user_type},
        "author_association": association,
        "body": body,
        "created_at": at,
        "updated_at": at,
        "html_url": f"https://github.com/octo/repo/pull/5#issuecomment-{cid}",
    }


TESTS = sa.Question("suite_green", "Suite passes?", "ci", options={"check": "tests"})
REPRO = sa.Question("repro_fixed", "Repro fixed?", "ci", options={"check": "repro"})


class ClientTests(unittest.TestCase):
    def test_headers_and_pagination(self):
        github, fake = gh(
            json_response(
                {"check_runs": [run("a")]},
                headers={"Link": '<https://api.github.com/next?page=2>; rel="next"'},
            ),
            json_response({"check_runs": [run("b", run_id=2)]}),
        )
        runs = github.check_runs("abc123")
        self.assertEqual([r["name"] for r in runs], ["a", "b"])
        self.assertEqual(fake.header("x-github-api-version", 0), "2026-03-10")
        self.assertEqual(fake.header("authorization", 0), "Bearer t0k")
        self.assertTrue(
            fake.requests[0].full_url.startswith(
                "https://api.github.com/repos/octo/repo/commits/abc123/check-runs?per_page=100"
            )
        )

    def test_report_ignores_look_alikes(self):
        github, fake = gh(
            json_response(
                [
                    comment(1, "bot", f"quoting {REPORT_MARKER}", user_type="Bot"),
                    comment(
                        2, "agent", f"{REPORT_MARKER}\nfake", association="COLLABORATOR"
                    ),
                ]
            ),
            json_response({"id": 3}),
        )
        post_report(github, 5, "report", exclude=["Agent"])
        self.assertEqual(fake.requests[1].get_method(), "POST")

    def test_a_report_too_long_for_a_comment_is_cut(self):
        github, fake = gh(json_response([]), json_response({"id": 3}))
        post_report(github, 5, "x" * 100_000)
        posted = fake.body(1)["body"]
        self.assertLessEqual(len(posted), 65_536, "GitHub's limit")
        self.assertTrue(posted.endswith("the run record has the rest)"))

    def test_report_is_updated_in_place(self):
        github, fake = gh(
            json_response(
                [comment(7, "bot", f"{REPORT_MARKER}\nold", user_type="Bot")]
            ),
            json_response({"id": 7}),
        )
        post_report(github, 5, "new report")
        self.assertEqual(fake.requests[1].get_method(), "PATCH")
        self.assertEqual(fake.body(1), {"body": f"{REPORT_MARKER}\nnew report"})


class ChecksTests(unittest.TestCase):
    def test_conclusions(self):
        github, _ = gh(
            json_response(
                {
                    "check_runs": [
                        run("tests", conclusion="failure", run_id=1),
                        run("tests", conclusion="success", run_id=2),
                        run("repro", conclusion="skipped", run_id=3),
                    ]
                }
            )
        )
        answers = asyncio.run(GitHubChecks(github)([TESTS, REPRO], context()))
        self.assertTrue(answers["suite_green"].is_met, "the latest run wins")
        self.assertFalse(
            answers["repro_fixed"].is_met, "a skipped check proves nothing"
        )
        self.assertEqual(answers["suite_green"].by, "ci:tests")
        self.assertIn(
            "https://github.com/octo/repo/runs/2", answers["suite_green"].evidence
        )

    def test_runs_in_progress_are_pending(self):
        github, _ = gh(
            json_response(
                {
                    "check_runs": [
                        run("tests"),
                        run("repro", status="in_progress", conclusion=None, run_id=2),
                    ]
                }
            )
        )
        with self.assertRaises(sa.Pending) as caught:
            asyncio.run(GitHubChecks(github)([TESTS, REPRO], context()))
        self.assertIn("in_progress", str(caught.exception))
        self.assertEqual(list(caught.exception.answers), ["suite_green"])

    def test_only_runs_from_the_trusted_app_count(self):
        github, _ = gh(
            json_response({"check_runs": [run("tests", app="other-ci", run_id=9)]})
        )
        with self.assertRaises(sa.Pending) as caught:
            asyncio.run(GitHubChecks(github)([TESTS], context()))
        self.assertIn(
            "no check run named 'tests' from github-actions", str(caught.exception)
        )

    def test_question_needs_a_check_name(self):
        github, _ = gh(json_response({"check_runs": []}))
        with self.assertRaisesRegex(ValueError, "needs check"):
            asyncio.run(GitHubChecks(github)([sa.Question("x", "X?", "ci")], context()))


SSO = sa.Question(
    "sso_exempt", "Is the SSO change fine?", "human", "YES if security signed off."
)
DOCS = sa.Question("docs_ok", "Are the docs right?", "human")
ASK = "<!-- symbolic-acceptance:ask ref=abc123 ids=sso_exempt,docs_ok -->\nquestions"


class ReviewsTests(unittest.TestCase):
    def test_first_call_posts_the_questions(self):
        github, fake = gh(json_response([]), json_response({"id": 1}))
        with self.assertRaises(sa.Pending):
            asyncio.run(Reviews(github, 5)([SSO], context()))
        posted = fake.body(1)["body"]
        self.assertIn(
            "<!-- symbolic-acceptance:ask ref=abc123 ids=sso_exempt -->", posted
        )
        self.assertIn("`sso_exempt`: Is the SSO change fine?", posted)
        self.assertIn("  - YES if security signed off.", posted)

    def test_reads_trusted_replies_after_the_question(self):
        comments = [
            comment(
                1,
                "early",
                "/yes sso_exempt before the question",
                at="2026-10-18T09:00:00Z",
            ),
            comment(2, "bot", ASK, user_type="Bot"),
            comment(
                3, "agent-bot[bot]", "/yes sso_exempt trust me", association="NONE"
            ),
            comment(4, "stranger", "/yes sso_exempt lgtm", association="NONE"),
            comment(
                5, "agent", "/yes docs_ok I wrote them", association="COLLABORATOR"
            ),
            comment(
                6,
                "alice",
                "/no sso_exempt needs security\n/yes docs_ok fine",
                at="2026-10-18T11:00:00Z",
            ),
            comment(
                7,
                "bob",
                "/YES sso_exempt: security signed off",
                at="2026-10-18T12:00:00Z",
            ),
            # A deleted account's comment has no user.
            dict(
                comment(8, "gone", "/no sso_exempt", at="2026-10-18T13:00:00Z"),
                user=None,
            ),
        ]
        github, _ = gh(json_response(comments))
        answers = asyncio.run(
            Reviews(github, 5, exclude=["agent"])([SSO, DOCS], context())
        )
        self.assertTrue(answers["sso_exempt"].is_met, "the latest reply wins")
        self.assertEqual(answers["sso_exempt"].by, "human:@bob")
        self.assertEqual(answers["sso_exempt"].reasoning, "security signed off")
        self.assertEqual(answers["docs_ok"].by, "human:@alice")

    def test_each_reply_that_may_not_answer_is_ignored(self):
        asked = comment(2, "bot", ASK, user_type="Bot")
        reply = "/yes sso_exempt ok"
        cases = {
            "a bot": comment(3, "helper", reply, user_type="Bot"),
            "a bot by its login": comment(3, "helper[bot]", reply),
            "the agent": comment(3, "Agent", reply, association="COLLABORATOR"),
            "a stranger": comment(3, "stranger", reply, association="NONE"),
            "an early reply": comment(3, "alice", reply, at="2026-10-18T09:00:00Z"),
            "an edited reply": dict(
                comment(3, "alice", reply), updated_at="2026-10-18T13:00:00Z"
            ),
            "a reply inside a line": comment(3, "alice", f"I'd say {reply}"),
        }
        for case, forbidden in cases.items():
            with self.subTest(case):
                github, _ = gh(json_response([asked, forbidden]))
                with self.assertRaises(sa.Pending) as caught:
                    asyncio.run(Reviews(github, 5, exclude=["agent"])([SSO], context()))
                self.assertEqual(caught.exception.answers, {})

    def test_an_edited_question_was_not_asked(self):
        asked = dict(
            comment(2, "bot", ASK, user_type="Bot"), updated_at="2026-10-18T13:00:00Z"
        )
        reply = comment(3, "alice", "/yes sso_exempt ok", at="2026-10-18T11:00:00Z")
        github, fake = gh(json_response([asked, reply]), json_response({"id": 9}))
        with self.assertRaises(sa.Pending) as caught:
            asyncio.run(Reviews(github, 5)([SSO], context()))
        self.assertEqual(caught.exception.answers, {})
        self.assertIn("ids=sso_exempt -->", fake.body(1)["body"], "asked afresh")

    def test_reviewer_allowlist_and_partial_answers(self):
        comments = [
            comment(2, "bot", ASK, user_type="Bot"),
            comment(3, "alice", "/yes sso_exempt ok"),
        ]
        github, _ = gh(json_response(comments))
        with self.assertRaises(sa.Pending) as caught:
            asyncio.run(Reviews(github, 5, reviewers=["Alice"])([SSO, DOCS], context()))
        self.assertEqual(list(caught.exception.answers), ["sso_exempt"])
        self.assertIn("docs_ok", str(caught.exception))

    def test_a_later_round_posts_only_its_new_questions(self):
        first = "<!-- symbolic-acceptance:ask ref=abc123 ids=sso_exempt -->\nquestions"
        comments = [
            comment(2, "bot", first, user_type="Bot"),
            comment(
                3,
                "alice",
                "/yes sso_exempt ok\n/yes docs_ok early",
                at="2026-10-18T10:30:00Z",
            ),
        ]
        github, fake = gh(json_response(comments), json_response({"id": 9}))
        with self.assertRaises(sa.Pending) as caught:
            asyncio.run(Reviews(github, 5)([SSO, DOCS], context()))
        posted = fake.body(1)["body"]
        self.assertIn("ids=docs_ok -->", posted)
        self.assertNotIn("`sso_exempt`", posted)
        # sso_exempt was asked before alice replied; docs_ok was not, so her
        # early "yes" to it does not count.
        self.assertEqual(list(caught.exception.answers), ["sso_exempt"])

    def test_only_the_runner_can_ask(self):
        fake_marker = "<!-- symbolic-acceptance:ask ref=abc123 ids=sso_exempt -->\nhi"
        comments = [
            comment(2, "agent", fake_marker, association="COLLABORATOR"),
            comment(3, "stranger", fake_marker, association="NONE"),
            comment(4, "bot", "Report:\n" + fake_marker, user_type="Bot"),
        ]
        github, fake = gh(json_response(comments), json_response({"id": 9}))
        with self.assertRaises(sa.Pending):
            asyncio.run(Reviews(github, 5, exclude=["agent"])([SSO], context()))
        self.assertIn("ids=sso_exempt -->", fake.body(1)["body"], "posted for real")

    def test_edited_comments_do_not_count(self):
        edited = comment(3, "alice", "/yes sso_exempt ok")
        edited["updated_at"] = "2026-10-18T13:00:00Z"
        github, _ = gh(json_response([comment(2, "bot", ASK, user_type="Bot"), edited]))
        with self.assertRaises(sa.Pending) as caught:
            asyncio.run(Reviews(github, 5)([SSO], context()))
        self.assertEqual(caught.exception.answers, {})

    def test_answers_for_another_commit_do_not_count(self):
        comments = [
            comment(2, "bot", ASK, user_type="Bot"),
            comment(3, "alice", "/yes sso_exempt ok"),
        ]
        github, fake = gh(json_response(comments), json_response({"id": 9}))
        with self.assertRaises(sa.Pending):
            asyncio.run(Reviews(github, 5)([SSO], context(ref="def456")))
        self.assertIn("ref=def456", fake.body(1)["body"])


if __name__ == "__main__":
    unittest.main()
