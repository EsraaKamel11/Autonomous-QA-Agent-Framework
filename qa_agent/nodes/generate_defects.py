"""
qa_agent/nodes/generate_defects.py
─────────────────────────────────────
DefectReporterAgent — node 10 of 12.

Decision boundary
─────────────────
Only answers: "How do I communicate these test findings to stakeholders?"
Never blocks merges, never routes the pipeline, never fixes the failing code.

Inputs consumed from state
──────────────────────────
  state["execution_report"]   ExecutionReport
  state["coverage_report"]    CoverageReport
  state["change_manifest"]    ChangeManifest
  state["pr_metadata"]        PRMetadata

State keys written
──────────────────
  defects               List[Defect]    — accumulated via operator.add
  jira_tickets_created  List[str]       — issue keys, accumulated manually
  current_phase         str
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from qa_agent.observability import log_node_execution
from qa_agent.prompts.defect_reporter import DEFECT_REPORTER_SYSTEM, DEFECT_REPORTER_USER
from qa_agent.state import (
    AgentError,
    CoverageReport,
    Defect,
    ExecutionReport,
    QAAgentState,
    Severity,
    TestResult,
)
from qa_agent.tools.jira_tools import JiraTicketResult, create_jira_defect

_VOCAREUM_API_KEY  = os.environ.get("VOCAREUM_API_KEY",  "voc-placeholder")
_VOCAREUM_BASE_URL = os.environ.get("VOCAREUM_BASE_URL", "https://openai.vocareum.com/v1")
_NODE_NAME = "generate_defects"

_JIRA_PROJECT = os.environ.get("JIRA_PROJECT", "QA")


# ──────────────────────────────────────────────────────────────────────────────
# LLM output model
# ──────────────────────────────────────────────────────────────────────────────

class _DefectItem(BaseModel):
    title:              str
    severity:           str             # critical | high | medium | low
    affected_endpoint:  str
    test_id:            str
    error_detail:       str
    reproduction_steps: List[str]       = Field(default_factory=list)


class _DefectReporterOutput(BaseModel):
    defects:    List[_DefectItem]
    pr_comment: str                     # full markdown PR comment


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _format_failed_tests_json(results: List[TestResult]) -> str:
    failed = [r for r in results if r.status in ("failed", "error")]
    if not failed:
        return "[]"
    return json.dumps(
        [
            {
                "test_id":       r.test_id,
                "scenario_id":   r.scenario_id,
                "status":        r.status,
                "failure_type":  r.failure_type.value if r.failure_type else None,
                "error_message": (r.error_message or "")[:500],
                "stdout":        (r.stdout or "")[:200],
            }
            for r in failed[:20]   # cap at 20 for prompt size
        ],
        indent=2,
    )


def _format_coverage_json(coverage: Optional[CoverageReport]) -> str:
    if not coverage:
        return "{}"
    return json.dumps(
        {
            "overall_score":         coverage.overall_score,
            "completeness_score":    coverage.completeness_score,
            "scenario_depth_score":  coverage.scenario_depth_score,
            "assertion_quality_score": coverage.assertion_quality_score,
            "regression_risk_score": coverage.regression_risk_score,
            "merge_recommendation":  coverage.merge_recommendation.value,
            "gaps":                  coverage.gaps[:10],
            "judge_reasoning":       coverage.judge_reasoning[:500],
        },
        indent=2,
    )


def _severity_from_str(raw: str) -> Severity:
    mapping = {
        "critical": Severity.CRITICAL,
        "high":     Severity.HIGH,
        "medium":   Severity.MEDIUM,
        "low":      Severity.LOW,
    }
    return mapping.get(raw.lower(), Severity.MEDIUM)


def _llm_item_to_defect(item: _DefectItem) -> Defect:
    return Defect(
        title=item.title[:200],
        severity=_severity_from_str(item.severity),
        affected_endpoint=item.affected_endpoint,
        test_id=item.test_id,
        error_detail=item.error_detail[:2000],
        reproduction_steps=item.reproduction_steps,
        jira_key=None,
    )


def _fallback_defects(results: List[TestResult]) -> List[Defect]:
    """Minimal defects when LLM is unavailable."""
    failed = [r for r in results if r.status in ("failed", "error")]
    defects = []
    for r in failed[:5]:
        defects.append(Defect(
            title=f"[FALLBACK] Test failure: {(r.test_id or 'unknown')[:60]}",
            severity=Severity.MEDIUM,
            affected_endpoint=r.scenario_id or "unknown",
            test_id=r.test_id or "unknown",
            error_detail=(r.error_message or r.stderr or "")[:500],
            reproduction_steps=[f"Run test: {r.test_id}"],
            jira_key=None,
        ))
    return defects


# ──────────────────────────────────────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────────────────────────────────────

def generate_defects(state: QAAgentState) -> dict:
    """
    DefectReporterAgent: for each test failure, generate a structured Defect
    record and create a Jira ticket.  Also generates the GitHub PR comment body
    (which finalize_node will post).

    Pipeline position: Node 10.
    Input:  state["execution_report"], state["coverage_report"],
            state["change_manifest"], state["pr_metadata"]
    Output: state["defects"] (operator.add), state["jira_tickets_created"],
            state["pr_comment_body"]

    LLM call: yes — with_structured_output(_DefectReporterOutput) via Vocareum.
    Tool calls: jira_tools.create_jira_defect per defect.
    """
    start_ts  = time.time()
    report    = state.get("execution_report")
    coverage  = state.get("coverage_report")
    manifest  = state.get("change_manifest")
    pr_meta   = state["pr_metadata"]
    errors:   list = []

    test_results = report.per_test_results if report else []

    # ── Build prompt context ──────────────────────────────────────────────────
    failed_json  = _format_failed_tests_json(test_results)
    coverage_json = _format_coverage_json(coverage)
    change_summary = manifest.change_summary if manifest else "(no change manifest)"

    pr_url = (
        f"https://github.com/{pr_meta.repo}/pull/{pr_meta.pr_number}"
    )

    user_message = DEFECT_REPORTER_USER.format(
        pr_number=pr_meta.pr_number,
        repo=pr_meta.repo,
        pr_url=pr_url,
        change_summary=change_summary[:2000],
        failed_tests_json=failed_json,
        coverage_report_json=coverage_json,
    )

    # ── LLM call ─────────────────────────────────────────────────────────────
    error_str: Optional[str] = None
    defects:   List[Defect]
    pr_comment_body: str

    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=_VOCAREUM_API_KEY,
            base_url=_VOCAREUM_BASE_URL,
            temperature=0,
        ).with_structured_output(_DefectReporterOutput)

        llm_output: _DefectReporterOutput = llm.invoke([
            {"role": "system", "content": DEFECT_REPORTER_SYSTEM},
            {"role": "user",   "content": user_message},
        ])

        defects        = [_llm_item_to_defect(item) for item in llm_output.defects]
        pr_comment_body = llm_output.pr_comment

    except Exception as exc:
        error_str = f"{type(exc).__name__}: {exc}"
        errors.append(AgentError(
            agent="DefectReporterAgent",
            phase="generate_defects",
            error=error_str,
            recoverable=True,
        ))
        defects         = _fallback_defects(test_results)
        pr_comment_body = (
            f"## QA Agent Report — PR #{pr_meta.pr_number}\n\n"
            f"**DefectReporterAgent LLM call failed.** "
            f"See decision log for details.\n\n"
            f"Failures: {len([r for r in test_results if r.status in ('failed','error')])}"
        )

    # ── Create Jira tickets ───────────────────────────────────────────────────
    jira_keys: List[str] = list(state.get("jira_tickets_created") or [])
    updated_defects: List[Defect] = []

    for defect in defects:
        try:
            ticket: JiraTicketResult = create_jira_defect(
                defect=defect,
                pr_url=pr_url,
                project_key=_JIRA_PROJECT,
            )
            jira_keys.append(ticket.issue_key)
            updated_defects.append(
                defect.model_copy(update={"jira_key": ticket.issue_key})
            )
        except Exception as jira_exc:
            # Jira failure is non-fatal — keep defect without jira_key
            updated_defects.append(defect)
            errors.append(AgentError(
                agent="DefectReporterAgent",
                phase="generate_defects.jira",
                error=str(jira_exc)[:200],
                recoverable=True,
            ))

    # ── Log before returning ──────────────────────────────────────────────────
    duration_ms = (time.time() - start_ts) * 1000
    severity_counts: dict = {}
    for d in updated_defects:
        k = d.severity.value
        severity_counts[k] = severity_counts.get(k, 0) + 1

    log_node_execution(
        node_name=_NODE_NAME,
        input_summary={
            "failed_tests":  len([r for r in test_results if r.status in ("failed","error")]),
            "coverage_score": coverage.overall_score if coverage else None,
        },
        output_summary={
            "defects_created":  len(updated_defects),
            "jira_tickets":     len(jira_keys),
            "by_severity":      str(severity_counts),
            "comment_length":   len(pr_comment_body),
        },
        routing_hint=(
            "human_review"
            if any(d.severity == Severity.CRITICAL for d in updated_defects)
            else "finalize"
        ),
        pr_number=pr_meta.pr_number,
        duration_ms=duration_ms,
        error=error_str,
    )

    result_dict: dict = {
        "defects":              updated_defects,   # operator.add
        "jira_tickets_created": jira_keys,
        "pr_comment_body":      pr_comment_body,
        "current_phase":        _NODE_NAME,
    }
    if errors:
        result_dict["errors"] = errors
    return result_dict
