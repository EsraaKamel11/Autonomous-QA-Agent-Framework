"""
qa_agent/nodes/evaluate_coverage.py
──────────────────────────────────────
CoverageEvaluatorAgent (LLM-as-Judge) — node 8 of 12.

Decision boundary
─────────────────
Only answers: "Is the test coverage sufficient to approve merging this PR?"
Never writes tests, never classifies failures, never routes the pipeline.

Inputs consumed from state
──────────────────────────
  state["execution_report"]   ExecutionReport
  state["change_manifest"]    ChangeManifest
  state["test_scenarios"]     List[TestScenario]

State keys written
──────────────────
  coverage_report   CoverageReport
  current_phase     str
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional

from langchain_openai import ChatOpenAI

from qa_agent.observability import log_node_execution
from qa_agent.prompts.coverage_judge import COVERAGE_JUDGE_SYSTEM, COVERAGE_JUDGE_USER
from qa_agent.state import (
    AgentError,
    CoverageReport,
    ExecutionReport,
    MergeRecommendation,
    QAAgentState,
    TestResult,
    TestScenario,
)

_VOCAREUM_API_KEY  = os.environ.get("VOCAREUM_API_KEY",  "voc-placeholder")
_VOCAREUM_BASE_URL = os.environ.get("VOCAREUM_BASE_URL", "https://openai.vocareum.com/v1")
_NODE_NAME = "evaluate_coverage"

# Weights for the four judge dimensions → overall_score
_WEIGHTS = {
    "completeness":      0.30,
    "scenario_depth":    0.25,
    "assertion_quality": 0.25,
    "regression_risk":   0.20,
}


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _format_execution_summary(report: ExecutionReport) -> str:
    return (
        f"Total:   {report.total}\n"
        f"Passed:  {report.passed}\n"
        f"Failed:  {report.failed}\n"
        f"Errors:  {report.errors}\n"
        f"Skipped: {report.skipped}\n"
        f"Duration: {report.duration_seconds:.1f}s"
    )


def _format_failed_tests(results: List[TestResult]) -> str:
    failed = [r for r in results if r.status in ("failed", "error")]
    if not failed:
        return "None — all tests passed."
    parts = []
    for r in failed[:10]:   # cap at 10 for prompt size
        parts.append(
            f"test_id: {r.test_id}\n"
            f"  status:        {r.status}\n"
            f"  failure_type:  {r.failure_type.value if r.failure_type else 'unknown'}\n"
            f"  error_message: {(r.error_message or '')[:200]}"
        )
    if len(failed) > 10:
        parts.append(f"... and {len(failed) - 10} more failures")
    return "\n\n".join(parts)


def _format_scenarios(scenarios: List[TestScenario]) -> str:
    if not scenarios:
        return "[]"
    return json.dumps(
        [
            {
                "scenario_id":  s.scenario_id,
                "endpoint":     s.endpoint,
                "method":       s.method,
                "type":         s.scenario_type.value,
                "description":  s.description,
                "priority":     s.priority,
            }
            for s in scenarios
        ],
        indent=2,
    )


def _format_affected_endpoints(change_manifest) -> str:
    if not change_manifest:
        return "[]"
    return json.dumps(
        [
            {
                "method":      ep.method,
                "path":        ep.path,
                "change_type": ep.change_type,
            }
            for ep in change_manifest.affected_endpoints
        ],
        indent=2,
    )


def _weighted_overall(report: CoverageReport) -> float:
    """Recompute overall_score from dimension scores using defined weights."""
    return round(
        report.completeness_score      * _WEIGHTS["completeness"]
        + report.scenario_depth_score  * _WEIGHTS["scenario_depth"]
        + report.assertion_quality_score * _WEIGHTS["assertion_quality"]
        + report.regression_risk_score * _WEIGHTS["regression_risk"],
        2,
    )


def _verdict_from_score(overall: float, report: CoverageReport) -> MergeRecommendation:
    """
    Apply the deterministic verdict rules as a safety override on the LLM verdict.
    The LLM verdict takes precedence unless it violates a hard rule.
    """
    if overall < 4.0:
        return MergeRecommendation.BLOCK
    if overall >= 7.0 and report.merge_recommendation == MergeRecommendation.APPROVE:
        return MergeRecommendation.APPROVE
    if overall >= 7.0:
        return MergeRecommendation.REQUEST_CHANGES
    return report.merge_recommendation  # trust LLM in the 4–7 range


def _fallback_report(report: Optional[ExecutionReport]) -> CoverageReport:
    """Deterministic fallback when the LLM call fails."""
    if report and report.total > 0:
        pass_rate = report.passed / report.total
        base_score = round(pass_rate * 6.0, 1)   # max 6 without LLM quality eval
    else:
        base_score = 0.0

    verdict = (
        MergeRecommendation.APPROVE if base_score >= 7.0
        else MergeRecommendation.REQUEST_CHANGES if base_score >= 4.0
        else MergeRecommendation.BLOCK
    )

    return CoverageReport(
        completeness_score=base_score,
        scenario_depth_score=base_score,
        assertion_quality_score=base_score,
        regression_risk_score=base_score,
        overall_score=base_score,
        gaps=["[FALLBACK] LLM-as-Judge unavailable — scores are heuristic only"],
        judge_reasoning="CoverageEvaluatorAgent LLM call failed; using pass-rate heuristic.",
        merge_recommendation=verdict,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_coverage(state: QAAgentState) -> dict:
    """
    CoverageEvaluatorAgent: score test coverage on 4 independent dimensions
    and produce a merge recommendation.

    Scoring dimensions (each 0–10):
      completeness      — are all changed endpoints covered?
      scenario_depth    — happy + edge + error cases present?
      assertion_quality — assertions verify behaviour, not just status codes?
      regression_risk   — high-risk changes covered more thoroughly?

    Verdict thresholds:
      APPROVE          overall_score ≥ 7.0
      REQUEST_CHANGES  overall_score 4.0–6.9
      BLOCK            overall_score < 4.0  OR  critical auth gap

    Pipeline position: Node 8.
    Input:  state["execution_report"], state["change_manifest"],
            state["test_scenarios"]
    Output: state["coverage_report"]

    LLM call: yes — with_structured_output(CoverageReport) via Vocareum.
    """
    start_ts = time.time()
    report   = state.get("execution_report")
    manifest = state.get("change_manifest")
    scenarios = state.get("test_scenarios") or []
    errors:  list = []

    # ── Build prompt context ──────────────────────────────────────────────────
    exec_summary  = _format_execution_summary(report) if report else "No execution data"
    failed_detail = _format_failed_tests(report.per_test_results if report else [])
    scenarios_txt = _format_scenarios(scenarios)
    endpoints_txt = _format_affected_endpoints(manifest)
    change_summary = manifest.change_summary if manifest else "(no change manifest)"
    risk_score     = manifest.risk_score if manifest else 0.5

    user_message = COVERAGE_JUDGE_USER.format(
        change_summary=change_summary[:2000],
        risk_score=risk_score,
        affected_endpoints=endpoints_txt,
        test_scenarios=scenarios_txt[:4000],
        execution_summary=exec_summary,
        failed_tests=failed_detail[:3000],
    )

    # ── LLM call ─────────────────────────────────────────────────────────────
    error_str: Optional[str] = None
    coverage: CoverageReport

    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=_VOCAREUM_API_KEY,
            base_url=_VOCAREUM_BASE_URL,
            temperature=0,
        ).with_structured_output(CoverageReport)

        coverage = llm.invoke([
            {"role": "system", "content": COVERAGE_JUDGE_SYSTEM},
            {"role": "user",   "content": user_message},
        ])

        # Recompute overall_score deterministically (don't trust LLM arithmetic)
        true_overall = _weighted_overall(coverage)
        true_verdict = _verdict_from_score(true_overall, coverage)
        coverage = coverage.model_copy(update={
            "overall_score":       true_overall,
            "merge_recommendation": true_verdict,
        })

    except Exception as exc:
        error_str = f"{type(exc).__name__}: {exc}"
        errors.append(AgentError(
            agent="CoverageEvaluatorAgent",
            phase="evaluate_coverage",
            error=error_str,
            recoverable=True,
        ))
        coverage = _fallback_report(report)

    # ── Routing hint for log ──────────────────────────────────────────────────
    aug_cycle = state.get("augmentation_cycle", 0)
    max_aug   = state.get("max_augmentation_cycles", 2)

    if coverage.overall_score < 6.0 and aug_cycle < max_aug:
        routing_hint = "augment_tests"
    elif coverage.merge_recommendation == MergeRecommendation.BLOCK:
        routing_hint = "human_review"
    else:
        routing_hint = "generate_defects"

    # ── Log before returning ──────────────────────────────────────────────────
    duration_ms = (time.time() - start_ts) * 1000
    log_node_execution(
        node_name=_NODE_NAME,
        input_summary={
            "total_tests":     report.total if report else 0,
            "passed":          report.passed if report else 0,
            "failed":          report.failed if report else 0,
            "scenarios":       len(scenarios),
            "risk_score":      risk_score,
        },
        output_summary={
            "overall_score":   coverage.overall_score,
            "completeness":    coverage.completeness_score,
            "scenario_depth":  coverage.scenario_depth_score,
            "assertion_quality": coverage.assertion_quality_score,
            "regression_risk": coverage.regression_risk_score,
            "verdict":         coverage.merge_recommendation.value,
            "gaps":            len(coverage.gaps),
        },
        routing_hint=routing_hint,
        pr_number=state["pr_metadata"].pr_number,
        duration_ms=duration_ms,
        error=error_str,
    )

    result: dict = {"coverage_report": coverage, "current_phase": _NODE_NAME}
    if errors:
        result["errors"] = errors
    return result
