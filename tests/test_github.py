import asyncio
import time
import unittest

from helpers import FakeHTTP, json_response, rendered

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
    github = GitHub("octo/repo", "t0k", "https://api.github.com", transport=fake)
    return github, fake


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
    html=None,
):
    """A comment, as GitHub lists it; ``html`` is how GitHub renders it,
    which plain lines are by default."""
    return {
        "id": cid,
        "user": {"login": login, "type": user_type},
        "author_association": association,
        "body": body,
        "body_html": html or rendered(body),
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
        self.assertEqual(fake.header("accept", 0), "application/vnd.github+json")
        self.assertTrue(
            fake.requests[0].full_url.startswith(
                "https://api.github.com/repos/octo/repo/commits/abc123/check-runs?per_page=100"
            )
        )
        # Comments come with the HTML GitHub renders them as.
        github, fake = gh(json_response([]))
        list(github.comments(5))
        self.assertEqual(fake.header("accept"), "application/vnd.github.full+json")

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
        report = post_report(github, 5, "new report")
        self.assertEqual(fake.requests[1].get_method(), "PATCH")
        self.assertEqual(fake.body(1), {"body": f"{REPORT_MARKER}\nnew report"})
        # Once known, it is updated without reading the comments again.
        github, fake = gh(json_response({"id": 7}))
        self.assertEqual(post_report(github, 5, "newer", report=report), 7)
        self.assertEqual([r.get_method() for r in fake.requests], ["PATCH"])

    def test_the_comments_are_read_only_until_the_report(self):
        report = comment(7, "bot", f"{REPORT_MARKER}\nold", user_type="Bot")
        first = json_response([report], headers={"link": '<https://x/2>; rel="next"'})
        github, fake = gh(first, json_response({"id": 7}))
        self.assertEqual(post_report(github, 5, "new"), 7)
        self.assertEqual([r.get_method() for r in fake.requests], ["GET", "PATCH"])


class ChecksTests(unittest.TestCase):
    def test_conclusions(self):
        github, _ = gh(
            json_response(
                {
                    "check_runs": [
                        run("tests", conclusion="failure", run_id=1),
                        run("tests", conclusion="success", run_id=2),
                        dict(
                            run("repro", conclusion="skipped", run_id=3),
                            output={"title": None, "summary": "Nothing to do"},
                        ),
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
        self.assertEqual(
            answers["suite_green"].evidence,
            ("https://github.com/octo/repo/runs/2", "tests success"),
        )
        self.assertEqual(
            answers["repro_fixed"].evidence,
            ("https://github.com/octo/repo/runs/3", "Nothing to do"),
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

    def test_runs_in_progress_are_read_again_while_waiting(self):
        going = run("tests", status="in_progress", conclusion=None)
        github, fake = gh(
            json_response({"check_runs": [going]}),
            json_response({"check_runs": [run("tests")]}),
        )
        checks = GitHubChecks(github, wait=60, poll=0)
        answers = asyncio.run(checks([TESTS], context()))
        self.assertTrue(answers["suite_green"].is_met)
        self.assertEqual(len(fake.requests), 2)
        # The last read is when the wait is up, however long the poll.
        github, fake = gh(*[json_response({"check_runs": [going]})] * 2)
        started = time.monotonic()
        with self.assertRaises(sa.Pending):
            asyncio.run(GitHubChecks(github, wait=0.2, poll=30)([TESTS], context()))
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(len(fake.requests), 2)

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
        for options in ({}, {"check": ""}, {"check": 3}):
            with self.subTest(options):
                github, _ = gh(json_response({"check_runs": []}))
                question = sa.Question("x", "X?", "ci", options=options)
                with self.assertRaisesRegex(ValueError, "needs check"):
                    asyncio.run(GitHubChecks(github)([question], context()))


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
            # The runner's bot asks; a bot counts whatever its association.
            comment(2, "github-actions[bot]", ASK, user_type="Bot", association="NONE"),
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

    def test_the_runners_own_comments_never_answer(self):
        # Run with a person's token, the runner posts as that person, and its
        # report quotes what the agent printed.
        report = f"{REPORT_MARKER}\n```text\n/yes sso_exempt security agreed\n```"
        comments = [
            comment(2, "alice", ASK),
            comment(3, "alice", report, at="2026-10-18T11:00:00Z"),
        ]
        github, _ = gh(json_response(comments))
        with self.assertRaises(sa.Pending) as caught:
            asyncio.run(Reviews(github, 5)([SSO], context()))
        self.assertEqual(caught.exception.answers, {})

    def test_a_reply_answers_the_latest_question_before_it(self):
        def ask(ref):
            return f"<!-- symbolic-acceptance:ask ref={ref} ids=sso_exempt -->\nq"

        comments = [
            comment(1, "bot", ask("aaa"), user_type="Bot", at="2026-10-18T10:00:00Z"),
            comment(
                2, "alice", "/no sso_exempt bypasses SSO", at="2026-10-18T10:10:00Z"
            ),
            comment(3, "bot", ask("bbb"), user_type="Bot", at="2026-10-18T11:00:00Z"),
            comment(4, "alice", "/yes sso_exempt fine now", at="2026-10-18T11:10:00Z"),
        ]
        # An attempt that goes back to aaa gets the answer given for aaa.
        for ref, is_met in (("aaa", False), ("bbb", True)):
            with self.subTest(ref):
                github, _ = gh(json_response(comments))
                answers = asyncio.run(Reviews(github, 5)([SSO], context(ref=ref)))
                self.assertIs(answers["sso_exempt"].is_met, is_met)

    def test_a_reply_posted_before_the_question_in_the_same_second_does_not_count(
        self,
    ):
        at = "2026-10-18T10:00:00Z"
        comments = [
            comment(11, "bot", ASK, user_type="Bot", at=at),
            comment(10, "alice", "/yes sso_exempt ok", at=at),
        ]
        github, _ = gh(json_response(comments))
        with self.assertRaises(sa.Pending) as caught:
            asyncio.run(Reviews(github, 5)([SSO], context()))
        self.assertEqual(caught.exception.answers, {})

    def test_what_github_shows_quoted_never_answers(self):
        # Replies are read from the comment as GitHub renders it, and what it
        # shows in a code block, in a list or not, or in a quote, may be the
        # agent's words; so may a line of inline code.
        block = (
            '<div class="snippet-clipboard-content notranslate position-relative'
            ' overflow-auto"><pre class="notranslate"><code>{}\n</code></pre></div>'
        )
        quoted = {
            "```\n/yes sso_exempt agreed\n```": block.format("/yes sso_exempt agreed"),
            "1. ```\n   pytest -k sso\n   ```\n2. The agent printed:\n```\n"
            "/yes sso_exempt agreed\n```": '<ol dir="auto">\n<li>'
            + block.format("pytest -k sso")
            + "</li>\n<li>The agent printed:</li>\n</ol>\n"
            + block.format("/yes sso_exempt agreed"),
            "> /yes sso_exempt agreed": "<blockquote>\n"
            '<p dir="auto">/yes sso_exempt agreed</p>\n</blockquote>',
            "`/yes sso_exempt agreed`": '<p dir="auto"><code class="notranslate">'
            "/yes sso_exempt agreed</code></p>",
        }
        for body, html in quoted.items():
            with self.subTest(body):
                comments = [
                    comment(2, "bot", ASK, user_type="Bot"),
                    comment(3, "alice", body, html=html),
                ]
                github, _ = gh(json_response(comments))
                with self.assertRaises(sa.Pending) as caught:
                    asyncio.run(Reviews(github, 5)([SSO], context()))
                self.assertEqual(caught.exception.answers, {})
        # Each block GitHub shows starts a line, however its HTML is laid out.
        comments = [
            comment(2, "bot", ASK, user_type="Bot"),
            comment(
                3,
                "alice",
                "Read it.\n\n/yes sso_exempt agreed",
                html='<p dir="auto">Read it.</p><p dir="auto">/yes sso_exempt agreed</p>',
            ),
        ]
        github, _ = gh(json_response(comments))
        answers = asyncio.run(Reviews(github, 5)([SSO], context()))
        self.assertTrue(answers["sso_exempt"].is_met)
        # What follows is the comment's own again, inline code and all.
        body = "```\n/yes sso_exempt agreed\n```\n/no sso_exempt `lock()` is gone"
        html = (
            block.format("/yes sso_exempt agreed")
            + '\n<p dir="auto">/no sso_exempt <code class="notranslate">lock()</code>'
            " is gone</p>"
        )
        comments = [
            comment(2, "bot", ASK, user_type="Bot"),
            comment(3, "alice", body, html=html),
        ]
        github, _ = gh(json_response(comments))
        answers = asyncio.run(Reviews(github, 5)([SSO], context()))
        self.assertEqual(
            (answers["sso_exempt"].is_met, answers["sso_exempt"].reasoning),
            (False, "`lock()` is gone"),
        )

    def test_a_reply_is_one_line(self):
        reply = comment(3, "alice", "Should I say\n/yes\nsso_exempt is fine? No idea")
        github, _ = gh(json_response([comment(2, "bot", ASK, user_type="Bot"), reply]))
        with self.assertRaises(sa.Pending) as caught:
            asyncio.run(Reviews(github, 5)([SSO], context()))
        self.assertEqual(caught.exception.answers, {})

    def test_logins_are_compared_as_github_does(self):
        reply = comment(3, "agent", "/yes sso_exempt", association="COLLABORATOR")
        # " agent" is what --exclude "alice, agent" gives.
        for exclude in ([" agent"], ["@Agent"]):
            with self.subTest(exclude):
                comments = [comment(2, "bot", ASK, user_type="Bot"), reply]
                github, _ = gh(json_response(comments))
                with self.assertRaises(sa.Pending) as caught:
                    asyncio.run(Reviews(github, 5, exclude=exclude)([SSO], context()))
                self.assertEqual(caught.exception.answers, {})

    def test_a_login_is_not_a_list_of_logins(self):
        # Read letter by letter, "agent" would exclude no one.
        github, _ = gh()
        for options in ({"exclude": "agent"}, {"reviewers": "alice"}):
            with self.subTest(options), self.assertRaisesRegex(TypeError, "list"):
                Reviews(github, 5, **options)

    def test_an_apps_account_is_excluded_with_or_without_its_bot_suffix(self):
        # The agent's app asks a question of its own, worded to mislead.
        forged = comment(2, "claude[bot]", ASK, association="NONE", user_type="Bot")
        reply = comment(
            3, "alice", "/yes sso_exempt typos fixed", at="2026-10-18T10:05:00Z"
        )
        for exclude in (["claude"], ["claude[bot]"]):
            with self.subTest(exclude):
                github, _ = gh(json_response([forged, reply]), json_response({"id": 4}))
                with self.assertRaises(sa.Pending) as caught:
                    asyncio.run(Reviews(github, 5, exclude=exclude)([SSO], context()))
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
