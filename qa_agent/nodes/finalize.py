"""
qa_agent/nodes/finalize.py
────────────────────────────
Pipeline finaliser — node 12 of 12.

Decision boundary
─────────────────
Publishes the results.  Posts the PR comment, sets the commit status, and
writes the final pipeline summary to the JSONL log.  Never makes merge
decisions — those were made by CoverageEvaluatorAgent and routing.py.

Inputs consumed from state
──────────────────────────
  state["coverage_report"]      CoverageReport
  state["defects"]              List[Defect]
  state["pr_metadata"]          PRMetadata
  state["pr_comment_body"]      str | None   — built by generate_defects
  state["generated_tests"]      List[TestScript]
  state["execution_report"]     ExecutionReport | None
  state["jira_tickets_created"] List[str]
  state["commit_status"]        str          — may be pre-set by human_review

State keys written
──────────────────
  pr_comment_body   str  — final markdown (may be fallback if missing)
  commit_status     str  — final GitHub commit status
  current_phase     str
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

from qa_agent.observability import log_node_execution
from qa_agent.prompts.defect_reporter import PR_COMMENT_TEMPLATE
from qa_agent.state import (
    AgentError,
    CoverageReport,
    Defect,
    ExecutionReport,
    MergeRecommendation,
    QAAgentState,
    Severity,
    TestScript,
)
from qa_agent.tools.github_tools import (
    CommitStatusResult,
    PRCommentResult,
    post_pr_comment,
    set_commit_status,
)

_NODE_NAME = "finalize"


# ──────────────────────────────────────────────────────────────────────────────
# Commit status mapping
# ──────────────────────────────────────────────────────────────────────────────

_VERDICT_TO_STATUS = {
    MergeRecommendation.APPROVE:         "success",
    MergeRecommendation.REQUEST_CHANGES: "failure",
    MergeRecommendation.BLOCK:           "failure",
}

_VERDICT_TO_DESCRIPTION = {
    MergeRecommendation.APPROVE:
        "QA Agent: All tests passed. Coverage sufficient. ✅",
    MergeRecommendation.REQUEST_CHANGES:
        "QA Agent: Test failures or coverage gaps detected. ⚠️",
    MergeRecommendation.BLOCK:
        "QA Agent: Critical failures or coverage gaps. Merge blocked. 🚫",
}

_VERDICT_BADGE = {
    MergeRecommendation.APPROVE:         "APPROVE ✅",
    MergeRecommendation.REQUEST_CHANGES: "REQUEST_CHANGES ⚠️",
    MergeRecommendation.BLOCK:           "BLOCK 🚫",
}


# ──────────────────────────────────────────────────────────────────────────────
# PR comment helpers
# ──────────────────────────────────────────────────────────────────────────────

def _format_defects_section(defects: List[Defect]) -> str:
    if not defects:
        return "_No defects filed._"
    lines = []
    icons = {
        Severity.CRITICAL: "🔴",
        Severity.HIGH:     "🟠",
        Severity.MEDIUM:   "🟡",
        Severity.LOW:      "🟢",
    }
    for d in defects:
        icon   = icons.get(d.severity, "⚪")
        jira   = f" — [{d.jira_key}]" if d.jira_key else ""
        lines.append(f"- {icon} **{d.severity.value.upper()}**{jira}: {d.title}")
    return "\n".join(lines)


def _format_gaps_section(coverage: Optional[CoverageReport]) -> str:
    if coverage is None or not coverage.gaps:
        return "_None identified._"
    return "\n".join(f"- {gap}" for gap in coverage.gaps[:10])


def _build_fallback_comment(
    pr_number:  int,
    coverage:   Optional[CoverageReport],
    defects:    List[Defect],
    report:     Optional[ExecutionReport],
    scripts:    List[TestScript],
) -> str:
    """
    Deterministic fallback comment built from state fields.
    Used when generate_defects failed to produce pr_comment_body.
    """
    verdict = (
        coverage.merge_recommendation
        if coverage else MergeRecommendation.REQUEST_CHANGES
    )

    pytest_count     = sum(1 for s in scripts if s.framework == "pytest")
    playwright_count = sum(1 for s in scripts if s.framework == "playwright")

    return PR_COMMENT_TEMPLATE.format(
        pr_number=pr_number,
        verdict_badge=_VERDICT_BADGE.get(verdict, "UNKNOWN"),
        completeness=  round(coverage.completeness_score, 1) if coverage else "N/A",
        scenario_depth=round(coverage.scenario_depth_score, 1) if coverage else "N/A",
        assertion_quality=round(coverage.assertion_quality_score, 1) if coverage else "N/A",
        regression_risk=round(coverage.regression_risk_score, 1) if coverage else "N/A",
        overall=round(coverage.overall_score, 1) if coverage else "N/A",
        passed=  report.passed  if report else 0,
        failed=  report.failed  if report else 0,
        errors=  report.errors  if report else 0,
        skipped= report.skipped if report else 0,
        defects_section=_format_defects_section(defects),
        gaps_section=_format_gaps_section(coverage),
        pytest_count=pytest_count,
        playwright_count=playwright_count,
    )


def _resolve_commit_status(
    state_status:  str,
    coverage:      Optional[CoverageReport],
) -> tuple[str, str]:
    """
    Determine final GitHub commit status and description string.

    Priority:
    1. If human_review set commit_status to "success" → honour it.
    2. If coverage_report has a verdict → use _VERDICT_TO_STATUS mapping.
    3. Fall back to the state's commit_status or "error".
    """
    # Human approved explicitly
    if state_status == "success":
        return "success", _VERDICT_TO_DESCRIPTION[MergeRecommendation.APPROVE]

    if coverage:
        verdict    = coverage.merge_recommendation
        gh_status  = _VERDICT_TO_STATUS.get(verdict, "error")
        description = _VERDICT_TO_DESCRIPTION.get(verdict, "QA Agent completed.")
        return gh_status, description

    # Fallback
    if state_status in ("pending", "success", "failure", "error"):
        return state_status, "QA Agent completed. See PR comment for details."

    return "error", "QA Agent encountered an unexpected state."


# ──────────────────────────────────────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────────────────────────────────────

def finalize(state: QAAgentState) -> dict:
    """
    Post the QA summary to GitHub and set the commit status.

    1. Determine final commit_status from coverage verdict (or human decision).
    2. Build or retrieve the PR comment markdown.
    3. Post the comment to GitHub via github_tools.post_pr_comment().
    4. Set the commit status via github_tools.set_commit_status().
    5. Write final pipeline summary to the JSONL decision log.

    Pipeline position: Node 12 — final node before END.
    Input:  state["coverage_report"], state["defects"], state["pr_metadata"],
            state["pr_comment_body"], state["generated_tests"],
            state["execution_report"], state["commit_status"]
    Output: state["pr_comment_body"], state["commit_status"]

    LLM calls: NONE.
    Tool calls: github_tools.post_pr_comment, github_tools.set_commit_status.
    """
    start_ts = time.time()
    pr_meta  = state["pr_metadata"]
    coverage = state.get("coverage_report")
    defects  = state.get("defects") or []
    report   = state.get("execution_report")
    scripts  = state.get("generated_tests") or []
    errors:  list = []

    # ── Step 1: Resolve commit status ─────────────────────────────────────────
    state_status  = state.get("commit_status", "pending")
    gh_status, gh_description = _resolve_commit_status(state_status, coverage)

    # ── Step 2: Build PR comment ──────────────────────────────────────────────
    pr_comment_body = state.get("pr_comment_body") or ""
    if not pr_comment_body.strip():
        # generate_defects didn't produce a comment — build fallback
        pr_comment_body = _build_fallback_comment(
            pr_number=pr_meta.pr_number,
            coverage=coverage,
            defects=defects,
            report=report,
            scripts=scripts,
        )

    # ── Step 3: Post PR comment ───────────────────────────────────────────────
    comment_result: Optional[PRCommentResult] = None
    try:
        comment_result = post_pr_comment(
            pr_number=pr_meta.pr_number,
            repo=pr_meta.repo,
            body=pr_comment_body,
        )
    except Exception as exc:
        errors.append(AgentError(
            agent="finalize",
            phase="post_pr_comment",
            error=str(exc)[:300],
            recoverable=True,
        ))

    # ── Step 4: Set commit status ─────────────────────────────────────────────
    status_result: Optional[CommitStatusResult] = None
    try:
        status_result = set_commit_status(
            repo=pr_meta.repo,
            sha=pr_meta.head_sha,
            state=gh_status,
            description=gh_description,
            context="qa-agent/autonomous-tests",
        )
    except Exception as exc:
        errors.append(AgentError(
            agent="finalize",
            phase="set_commit_status",
            error=str(exc)[:300],
            recoverable=True,
        ))

    # ── Step 5: Log final pipeline summary ───────────────────────────────────
    duration_ms = (time.time() - start_ts) * 1000
    log_node_execution(
        node_name=_NODE_NAME,
        input_summary={
            "pr_number":      pr_meta.pr_number,
            "verdict":        coverage.merge_recommendation.value if coverage else "N/A",
            "overall_score":  coverage.overall_score if coverage else None,
            "defects":        len(defects),
            "jira_tickets":   len(state.get("jira_tickets_created") or []),
        },
        output_summary={
            "commit_status":   gh_status,
            "comment_posted":  comment_result is not None,
            "comment_url":     comment_result.url if comment_result else None,
            "status_set":      status_result.success if status_result else False,
        },
        routing_hint="END",
        pr_number=pr_meta.pr_number,
        duration_ms=duration_ms,
        error=(errors[0].error if errors else None),
    )

    result: dict = {
        "pr_comment_body": pr_comment_body,
        "commit_status":   gh_status,
        "current_phase":   _NODE_NAME,
    }
    if errors:
        result["errors"] = errors
    return result
