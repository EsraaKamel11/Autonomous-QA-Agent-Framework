"""
qa_agent/tools/jira_tools.py
─────────────────────────────
Jira integration tools for defect ticket creation and status management.

Decision boundary
─────────────────
These tools answer only:
  • "Create a Jira bug ticket with these details."
  • "Transition this Jira issue to a new status."

They never classify defects, determine severity, or decide which failures
are worth filing. All of that belongs to DefectReporterAgent, which builds
the ticket payload before calling these functions.

Live vs. mock
─────────────
Set  USE_LIVE_JIRA=true  to enable real python-jira calls.
Leave unset (or anything else) to use mock mode — no Jira credentials
needed. Mock returns realistic-looking ticket keys and URLs.

Required env vars when USE_LIVE_JIRA=true
─────────────────────────────────────────
  JIRA_SERVER   Base URL of your Jira instance, e.g. https://myorg.atlassian.net
  JIRA_EMAIL    Atlassian account email for basic auth.
  JIRA_TOKEN    Atlassian API token (NOT your password).
  JIRA_PROJECT  Default project key, e.g. "QA" or "BUG".
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Optional

from pydantic import BaseModel

from qa_agent.state import Defect, Severity


# ──────────────────────────────────────────────────────────────────────────────
# Local result models
# ──────────────────────────────────────────────────────────────────────────────

class JiraTicketResult(BaseModel):
    """
    Confirmation of a created or updated Jira issue.
    Returned by create_jira_defect(); the jira_key is written back into the
    Defect model by the generate_defects node.
    """
    issue_key:    str           # e.g. "QA-42"
    url:          str           # e.g. "https://myorg.atlassian.net/browse/QA-42"
    project:      str
    summary:      str
    status:       str           # Jira status at creation time (usually "Open")
    created:      bool          # True = new ticket; False = existing ticket updated


class JiraTransitionResult(BaseModel):
    """Result of a status-transition operation on an existing Jira issue."""
    issue_key:    str
    from_status:  str
    to_status:    str
    success:      bool


# ──────────────────────────────────────────────────────────────────────────────
# Feature flag + constants
# ──────────────────────────────────────────────────────────────────────────────

_USE_LIVE = os.getenv("USE_LIVE_JIRA", "").lower() == "true"

# Maps our Severity enum to Jira priority names
_PRIORITY_MAP: dict[str, str] = {
    Severity.CRITICAL: "Highest",
    Severity.HIGH:     "High",
    Severity.MEDIUM:   "Medium",
    Severity.LOW:      "Low",
}

# Jira status transitions most orgs have in a standard Bug workflow
_COMMON_TRANSITIONS: dict[str, str] = {
    "In Progress":  "21",
    "Done":         "31",
    "Closed":       "31",
    "Reopened":     "51",
    "In Review":    "41",
}

_DEFAULT_PROJECT = os.environ.get("JIRA_PROJECT", "QA")


# ──────────────────────────────────────────────────────────────────────────────
# Mock helpers
# ──────────────────────────────────────────────────────────────────────────────

_mock_counter = 100   # mutable module-level counter for realistic-looking keys


def _mock_issue_key(project: str, summary: str) -> str:
    """Generate a deterministic-ish fake Jira issue key."""
    global _mock_counter
    _mock_counter += 1
    return f"{project}-{_mock_counter}"


def _mock_url(server: str, key: str) -> str:
    base = server.rstrip("/") if server else "https://mock.atlassian.net"
    return f"{base}/browse/{key}"


def _build_description(defect: Defect, pr_url: str) -> str:
    """
    Construct a Jira description from a Defect model.

    Uses Jira wiki markup (not Markdown) for legacy Jira; Jira Cloud also
    accepts Atlassian Document Format but wiki markup is universally safe.
    """
    steps = "\n".join(
        f"# {i + 1}. {step}" for i, step in enumerate(defect.reproduction_steps)
    )
    return (
        f"h2. Summary\n"
        f"{defect.error_detail}\n\n"
        f"h2. Affected Endpoint\n"
        f"{{code}}{defect.affected_endpoint}{{code}}\n\n"
        f"h2. Test ID\n"
        f"{{code}}{defect.test_id}{{code}}\n\n"
        f"h2. Reproduction Steps\n"
        f"{steps or '# See QA Agent decision log for details.'}\n\n"
        f"h2. Pull Request\n"
        f"{pr_url}\n\n"
        f"_Automatically created by QA Agent._"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public tool functions
# ──────────────────────────────────────────────────────────────────────────────

def create_jira_defect(
    defect:      Defect,
    pr_url:      str,
    project_key: str = _DEFAULT_PROJECT,
    labels:      Optional[List[str]] = None,
) -> JiraTicketResult:
    """
    Create a Jira bug ticket for a test failure.

    Decision boundary: creates exactly the ticket described by the Defect
    model it receives. Never decides severity, priority, or project — those
    are set by DefectReporterAgent before this function is called.

    Parameters
    ----------
    defect      : Defect    Structured defect from DefectReporterAgent.
    pr_url      : str       Full URL of the GitHub PR (included in ticket body).
    project_key : str       Jira project key to create the issue in.
                            Falls back to JIRA_PROJECT env var, then "QA".
    labels      : list[str] Additional labels to apply beyond ["qa-agent"].

    Returns
    -------
    JiraTicketResult
        Issue key, URL, and creation metadata.
    """
    base_labels = ["qa-agent", f"severity-{defect.severity.value}"]
    if labels:
        base_labels.extend(labels)

    summary = f"[QA-Agent] {defect.title}"
    description = _build_description(defect, pr_url)
    priority = _PRIORITY_MAP.get(defect.severity, "Medium")

    if not _USE_LIVE:
        key = _mock_issue_key(project_key, summary)
        server = os.environ.get("JIRA_SERVER", "https://mock.atlassian.net")
        return JiraTicketResult(
            issue_key=key,
            url=_mock_url(server, key),
            project=project_key,
            summary=summary,
            status="Open",
            created=True,
        )

    from jira import JIRA  # imported lazily — only needed in live mode

    client = JIRA(
        server=os.environ["JIRA_SERVER"],
        basic_auth=(os.environ["JIRA_EMAIL"], os.environ["JIRA_TOKEN"]),
    )

    fields: dict = {
        "project":     {"key": project_key},
        "summary":     summary,
        "description": description,
        "issuetype":   {"name": "Bug"},
        "priority":    {"name": priority},
        "labels":      base_labels,
    }

    issue = client.create_issue(fields=fields)

    return JiraTicketResult(
        issue_key=str(issue.key),
        url=f"{os.environ['JIRA_SERVER'].rstrip('/')}/browse/{issue.key}",
        project=project_key,
        summary=summary,
        status=str(issue.fields.status.name),
        created=True,
    )


def update_jira_status(
    issue_key: str,
    new_status: str,
) -> JiraTransitionResult:
    """
    Transition a Jira issue to a new workflow status.

    Decision boundary: executes the transition requested. Never decides
    which status is appropriate — that is the finalize node's responsibility.

    Parameters
    ----------
    issue_key  : str   Jira issue key, e.g. "QA-42".
    new_status : str   Target status name, e.g. "In Progress", "Done".

    Returns
    -------
    JiraTransitionResult
        Before/after status and success flag.
    """
    if not _USE_LIVE:
        return JiraTransitionResult(
            issue_key=issue_key,
            from_status="Open",
            to_status=new_status,
            success=True,
        )

    from jira import JIRA

    client = JIRA(
        server=os.environ["JIRA_SERVER"],
        basic_auth=(os.environ["JIRA_EMAIL"], os.environ["JIRA_TOKEN"]),
    )

    issue = client.issue(issue_key)
    from_status = str(issue.fields.status.name)

    # Find the matching transition by name (case-insensitive)
    transitions = client.transitions(issue)
    match = next(
        (t for t in transitions if t["name"].lower() == new_status.lower()),
        None,
    )

    if match is None:
        return JiraTransitionResult(
            issue_key=issue_key,
            from_status=from_status,
            to_status=new_status,
            success=False,
        )

    client.transition_issue(issue, match["id"])

    return JiraTransitionResult(
        issue_key=issue_key,
        from_status=from_status,
        to_status=new_status,
        success=True,
    )


def add_jira_comment(
    issue_key: str,
    comment:   str,
) -> bool:
    """
    Append a plain-text or wiki-markup comment to an existing Jira issue.

    Decision boundary: posts exactly the comment it receives. Never formats
    or summarises the comment content.

    Returns
    -------
    bool   True on success; False on failure (never raises in mock mode).
    """
    if not _USE_LIVE:
        return True

    from jira import JIRA

    try:
        client = JIRA(
            server=os.environ["JIRA_SERVER"],
            basic_auth=(os.environ["JIRA_EMAIL"], os.environ["JIRA_TOKEN"]),
        )
        client.add_comment(issue_key, comment)
        return True
    except Exception:
        return False


def build_defects_from_results(
    failed_tests:      List[dict],
    coverage_gaps:     List[str],
    pr_number:         int,
    affected_endpoint: str = "unknown",
) -> List[Defect]:
    """
    Construct a list of Defect models from raw test failure data.

    This is a pure data-transformation helper — no I/O, no LLM calls.
    DefectReporterAgent calls this before calling create_jira_defect().

    Decision boundary: maps failure data to Defect models only. Never
    decides what to do with the defects — DefectReporterAgent does that.

    Parameters
    ----------
    failed_tests      : list[dict]  Raw failure records from ExecutionReport.
    coverage_gaps     : list[str]   Gap descriptions from CoverageReport.
    pr_number         : int         PR number for traceability.
    affected_endpoint : str         Endpoint associated with failures.

    Returns
    -------
    List[Defect]   One Defect per failed test (coverage gaps produce no
                   individual Defects — they become a single summary ticket).
    """
    defects: List[Defect] = []

    for test in failed_tests:
        status    = test.get("status", "failed")
        error_msg = test.get("error_message") or test.get("longrepr", "")
        test_id   = test.get("test_id") or test.get("node_id", "unknown")

        # Rough severity heuristic — DefectReporterAgent may override via LLM
        if "auth" in error_msg.lower() or "403" in error_msg or "401" in error_msg:
            severity = Severity.CRITICAL
        elif "500" in error_msg or "schema" in error_msg.lower():
            severity = Severity.HIGH
        elif status == "error":
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM

        defects.append(
            Defect(
                title=f"Test failure: {test_id}",
                severity=severity,
                affected_endpoint=affected_endpoint,
                test_id=test_id,
                error_detail=error_msg[:2000],
                reproduction_steps=[
                    f"Run PR #{pr_number} QA pipeline.",
                    f"Execute test: {test_id}",
                    f"Observe failure: {error_msg[:300]}",
                ],
                jira_key=None,
            )
        )

    return defects
