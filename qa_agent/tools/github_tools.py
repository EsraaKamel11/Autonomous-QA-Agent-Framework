"""
qa_agent/tools/github_tools.py
───────────────────────────────
GitHub API integration tools for PR inspection and status reporting.

Decision boundary
─────────────────
These tools answer only:
  • "What files changed in this PR?"
  • "What does the diff look like?"
  • "Post this comment / set this status."

They never interpret what changed, estimate risk, or decide what to do next.
All of that belongs to CodeAnalystAgent and SupervisorAgent.

Live vs. mock
─────────────
Set  USE_LIVE_GITHUB=true  to enable real PyGithub calls.
Leave unset (or set to anything else) to use realistic mock data — useful
for local development and CI runs without a GITHUB_TOKEN.

Required env vars when USE_LIVE_GITHUB=true
───────────────────────────────────────────
  GITHUB_TOKEN   Personal-access-token or Actions GITHUB_TOKEN with
                 repo read + statuses write + pull_requests write scopes.
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Optional

from pydantic import BaseModel


# ──────────────────────────────────────────────────────────────────────────────
# Local result models
# (These represent GitHub API responses — not QA pipeline state. They are kept
#  here rather than state.py because they don't flow through LangGraph state.)
# ──────────────────────────────────────────────────────────────────────────────

class ChangedFile(BaseModel):
    """One file entry from the GitHub PR files API."""
    filename:  str
    status:    str          # added | removed | modified | renamed | copied
    additions: int
    deletions: int
    patch:     Optional[str] = None   # unified diff for this file; None for binary


class PRCommentResult(BaseModel):
    """Result of a successful PR comment post."""
    comment_id:   int
    url:          str
    body_preview: str       # first 120 chars of the posted body


class CommitStatusResult(BaseModel):
    """Result of a GitHub commit-status write."""
    repo:        str
    sha:         str
    state:       str        # pending | success | failure | error
    description: str
    context:     str
    success:     bool


# ──────────────────────────────────────────────────────────────────────────────
# Feature flag
# ──────────────────────────────────────────────────────────────────────────────

_USE_LIVE = os.getenv("USE_LIVE_GITHUB", "").lower() == "true"


# ──────────────────────────────────────────────────────────────────────────────
# Mock helpers — realistic data that matches production schemas exactly
# ──────────────────────────────────────────────────────────────────────────────

def _mock_diff(pr_number: int, repo: str) -> str:
    """
    Return a realistic unified diff representing an API endpoint change.
    The mock shows a GET /users/{id} endpoint gaining structured error
    responses and a new response schema — a common real-world pattern.
    """
    return (
        f"diff --git a/api/users.py b/api/users.py\n"
        f"index 3e4f5a..7b8c9d 100644\n"
        f"--- a/api/users.py\n"
        f"+++ b/api/users.py\n"
        f"@@ -1,5 +1,6 @@\n"
        f" from flask import jsonify, request\n"
        f" from models import User, db\n"
        f"+from schemas import UserSchema\n"
        f" \n"
        f" \n"
        f" def get_user(user_id: int):\n"
        f"@@ -10,7 +11,10 @@ def get_user(user_id: int):\n"
        f"     user = db.session.get(User, user_id)\n"
        f"-    return jsonify(user.to_dict()) if user else ('', 404)\n"
        f"+    if not user:\n"
        f"+        return jsonify({{\"error\": \"User not found\", "
        f"\"code\": \"USER_NOT_FOUND\"}}), 404\n"
        f"+    schema = UserSchema()\n"
        f"+    return jsonify(schema.dump(user)), 200\n"
        f" \n"
        f"diff --git a/api/schemas.py b/api/schemas.py\n"
        f"index 0000000..1a2b3c4\n"
        f"--- /dev/null\n"
        f"+++ b/api/schemas.py\n"
        f"@@ -0,0 +1,12 @@\n"
        f"+from marshmallow import Schema, fields\n"
        f"+\n"
        f"+\n"
        f"+class UserSchema(Schema):\n"
        f"+    id         = fields.Int(dump_only=True)\n"
        f"+    username   = fields.Str(required=True)\n"
        f"+    email      = fields.Email(required=True)\n"
        f"+    created_at = fields.DateTime(dump_only=True)\n"
        f"+    is_active  = fields.Bool(load_default=True)\n"
        f"\n"
        f"# [mock diff — PR #{pr_number} in {repo}]\n"
    )


def _mock_changed_files() -> List[ChangedFile]:
    """
    Return a realistic list of changed files matching the mock diff above.
    """
    return [
        ChangedFile(
            filename="api/users.py",
            status="modified",
            additions=18,
            deletions=5,
            patch=(
                "@@ -10,7 +11,10 @@ def get_user(user_id: int):\n"
                "     user = db.session.get(User, user_id)\n"
                "-    return jsonify(user.to_dict()) if user else ('', 404)\n"
                "+    if not user:\n"
                "+        return jsonify({\"error\": \"User not found\"}), 404\n"
                "+    return jsonify(UserSchema().dump(user)), 200\n"
            ),
        ),
        ChangedFile(
            filename="api/schemas.py",
            status="added",
            additions=12,
            deletions=0,
            patch=(
                "@@ -0,0 +1,12 @@\n"
                "+class UserSchema(Schema):\n"
                "+    id       = fields.Int(dump_only=True)\n"
                "+    username = fields.Str(required=True)\n"
                "+    email    = fields.Email(required=True)\n"
            ),
        ),
        ChangedFile(
            filename="tests/test_users.py",
            status="modified",
            additions=34,
            deletions=9,
            patch=(
                "@@ -1,4 +1,8 @@\n"
                " import pytest\n"
                "+from api.schemas import UserSchema\n"
            ),
        ),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Public tool functions
# ──────────────────────────────────────────────────────────────────────────────

def get_pr_diff(pr_number: int, repo: str) -> str:
    """
    Fetch the full unified diff for every file in the pull request.

    Decision boundary: returns raw text only. Does NOT classify, score,
    or derive anything from the diff — CodeAnalystAgent does that.

    Parameters
    ----------
    pr_number : int   GitHub PR number (e.g. 142).
    repo      : str   Full repo slug "owner/repo" (e.g. "acme/payments-api").

    Returns
    -------
    str
        Concatenated unified diff across all changed files.
        Empty string if the PR has no changed files.
    """
    if not _USE_LIVE:
        return _mock_diff(pr_number, repo)

    from github import Github  # imported lazily — only needed in live mode

    g = Github(os.environ["GITHUB_TOKEN"])
    repo_obj = g.get_repo(repo)
    pr = repo_obj.get_pull(pr_number)

    parts: List[str] = []
    for f in pr.get_files():
        parts.append(
            f"diff --git a/{f.filename} b/{f.filename}\n"
            f"--- a/{f.filename}\n"
            f"+++ b/{f.filename}\n"
            f"{f.patch or ''}"
        )
    return "\n".join(parts)


def get_changed_files(pr_number: int, repo: str) -> List[ChangedFile]:
    """
    List every file touched by the PR with change-type metadata.

    Decision boundary: returns file-level metadata only. Does NOT derive
    which API endpoints were affected — CodeAnalystAgent does that by
    correlating filenames with the OpenAPI spec paths.

    Parameters
    ----------
    pr_number : int   GitHub PR number.
    repo      : str   Full repo slug "owner/repo".

    Returns
    -------
    List[ChangedFile]
        One entry per changed file, sorted by filename.
    """
    if not _USE_LIVE:
        return _mock_changed_files()

    from github import Github

    g = Github(os.environ["GITHUB_TOKEN"])
    pr = g.get_repo(repo).get_pull(pr_number)
    return sorted(
        [
            ChangedFile(
                filename=f.filename,
                status=f.status,
                additions=f.additions,
                deletions=f.deletions,
                patch=f.patch or None,
            )
            for f in pr.get_files()
        ],
        key=lambda c: c.filename,
    )


def post_pr_comment(pr_number: int, repo: str, body: str) -> PRCommentResult:
    """
    Post a markdown comment to a GitHub PR.

    Decision boundary: posts exactly the body it receives. Never summarises,
    reformats, or truncates the body — DefectReporterAgent is responsible for
    building the final markdown string before calling this function.

    Parameters
    ----------
    pr_number : int   GitHub PR number.
    repo      : str   Full repo slug "owner/repo".
    body      : str   Markdown body; may contain code fences and tables.

    Returns
    -------
    PRCommentResult
        comment_id and url of the created comment.
    """
    preview = body[:120].replace("\n", " ")

    if not _USE_LIVE:
        fake_id = int(
            hashlib.md5(f"{pr_number}{repo}{body[:32]}".encode()).hexdigest()[:8], 16
        )
        return PRCommentResult(
            comment_id=fake_id,
            url=(
                f"https://github.com/{repo}/pull/{pr_number}"
                f"#issuecomment-{fake_id}"
            ),
            body_preview=preview,
        )

    from github import Github

    g = Github(os.environ["GITHUB_TOKEN"])
    pr = g.get_repo(repo).get_pull(pr_number)
    comment = pr.create_issue_comment(body)
    return PRCommentResult(
        comment_id=comment.id,
        url=comment.html_url,
        body_preview=preview,
    )


def set_commit_status(
    repo: str,
    sha: str,
    state: str,
    description: str,
    context: str = "qa-agent/autonomous-tests",
) -> CommitStatusResult:
    """
    Set the GitHub commit status that controls PR mergeability.

    Decision boundary: writes exactly the state/description provided.
    Never decides what state to set — that is the finalize node's job,
    which reads the CoverageReport.merge_recommendation.

    Parameters
    ----------
    repo        : str   Full repo slug "owner/repo".
    sha         : str   Full 40-char commit SHA (head_sha from PRMetadata).
    state       : str   One of: pending | success | failure | error
    description : str   Short human-readable status (≤ 140 chars enforced).
    context     : str   Status check name shown in the GitHub PR UI.

    Returns
    -------
    CommitStatusResult
        Echo of all written fields plus a success flag.
    """
    # GitHub enforces 140-char limit on description
    description = description[:140]

    if not _USE_LIVE:
        return CommitStatusResult(
            repo=repo,
            sha=sha,
            state=state,
            description=description,
            context=context,
            success=True,
        )

    from github import Github

    g = Github(os.environ["GITHUB_TOKEN"])
    repo_obj = g.get_repo(repo)
    commit = repo_obj.get_commit(sha)
    commit.create_status(
        state=state,
        description=description,
        context=context,
    )
    return CommitStatusResult(
        repo=repo,
        sha=sha,
        state=state,
        description=description,
        context=context,
        success=True,
    )
