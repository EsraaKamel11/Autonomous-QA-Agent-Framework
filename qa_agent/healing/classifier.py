"""
qa_agent/healing/classifier.py
────────────────────────────────
Two-stage test-failure classifier for the SelfHealingAgent.

Stage 1 — Rule-based (fast, zero LLM tokens)
    Applies regex patterns with per-pattern confidence weights against the
    combined error text.  Returns a winner immediately if its accumulated
    confidence ≥ _LLM_THRESHOLD (default 0.6).

Stage 2 — LLM fallback (used only when Stage 1 confidence < 0.6)
    Sends the error text to GPT-4o via the Vocareum proxy using
    with_structured_output() — never parses raw JSON strings.

Output
──────
ClassificationResult dataclass:
    failure_type   : FailureType  — one of ASSERTION, SCHEMA, SELECTOR,
                                    ENVIRONMENT, AUTH
    confidence     : float        — 0.0–1.0
    method         : str          — "rule_based" | "llm_fallback"
    reasoning      : str          — human-readable explanation
    matched_patterns: list[str]   — regex patterns that fired (rule_based only)

Auto-healable check
───────────────────
is_auto_healable(failure_type) → True only for ASSERTION, SCHEMA, SELECTOR.
ENVIRONMENT and AUTH always escalate to human_review immediately.

Design notes
────────────
- FailureType.LOGIC (from state.py) is intentionally never returned here.
  The classifier's contract is the 5 types listed above.  LOGIC would
  represent "real bug found" and is handled by the CoverageEvaluatorAgent
  path, not the self-healing path.

- Pattern weights express individual diagnostic power, not combined
  probability.  A single strong pattern (weight 0.90) is enough to pass
  the threshold without needing corroboration.

- When multiple types score above the threshold, the highest-scoring type
  wins.  True ties (rounding to same float) go to the type with the higher
  average individual weight.

Vocareum credentials
────────────────────
    VOCAREUM_API_KEY   — Vocareum OpenAI proxy key  (starts with "voc-")
    VOCAREUM_BASE_URL  — proxy base URL
                         (default: https://openai.vocareum.com/v1)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from qa_agent.state import FailureType

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

_VOCAREUM_API_KEY  = os.environ.get("VOCAREUM_API_KEY",  "voc-placeholder")
_VOCAREUM_BASE_URL = os.environ.get("VOCAREUM_BASE_URL", "https://openai.vocareum.com/v1")

# Stage 1 → Stage 2 handoff threshold
_LLM_THRESHOLD: float = 0.6

# Types that SelfHealingAgent can repair without human intervention
_AUTO_HEALABLE: frozenset[FailureType] = frozenset({
    FailureType.ASSERTION,
    FailureType.SCHEMA,
    FailureType.SELECTOR,
})


# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """
    Output of classify_failure().

    Attributes
    ----------
    failure_type     : One of the 5 healable/non-healable FailureType values.
    confidence       : Accumulated confidence in [0.0, 1.0].
                       Rule-based: sum of matched weights, capped at 1.0.
                       LLM fallback: the model's self-reported confidence.
    method           : "rule_based" or "llm_fallback".
    reasoning        : Human-readable explanation suitable for the JSONL log.
    matched_patterns : Regex patterns that contributed to the score
                       (empty list for llm_fallback results).
    """
    failure_type:     FailureType
    confidence:       float
    method:           str
    reasoning:        str
    matched_patterns: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Pattern rule definition
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PatternRule:
    """
    A single diagnostic regex rule with associated confidence weight.

    Attributes
    ----------
    pattern          : Raw regex string (passed to re.search).
    weight           : Confidence contribution when this pattern matches.
                       Range (0, 1].  A single match can raise the type's
                       accumulated score up to this weight.
    case_insensitive : Apply re.IGNORECASE (default True).
    description      : Human label — appears in matched_patterns output.
    """
    pattern:          str
    weight:           float
    case_insensitive: bool = True
    description:      str  = ""


# ──────────────────────────────────────────────────────────────────────────────
# Rule tables — one list per FailureType
# ──────────────────────────────────────────────────────────────────────────────
#
# Weight guidelines
# -----------------
# 0.90–0.95  Unambiguous signal: the pattern can only appear for this type.
# 0.70–0.89  Strong signal: almost always this type, rare false positives.
# 0.50–0.69  Moderate signal: meaningful contribution but needs corroboration.
# 0.20–0.49  Weak signal: requires other patterns to cross the threshold.
#
# Note: ENVIRONMENT and SELECTOR both have "timeout" in their vocabulary.
# The ENVIRONMENT timeout patterns are generic ("timed out", "timeout error").
# The SELECTOR timeout pattern is specific ("waiting for selector/locator").
# Order of matching matters less than weight specificity.

_RULES: Dict[FailureType, List[PatternRule]] = {

    # ── AUTH ──────────────────────────────────────────────────────────────────
    FailureType.AUTH: [
        PatternRule(
            r"\b401\b", 0.90,
            description="HTTP 401 Unauthorized status code",
        ),
        PatternRule(
            r"\b403\b", 0.90,
            description="HTTP 403 Forbidden status code",
        ),
        PatternRule(
            r"\bunauthorized\b", 0.90,
            description="'Unauthorized' text in error",
        ),
        PatternRule(
            r"\bforbidden\b", 0.90,
            description="'Forbidden' text in error",
        ),
        PatternRule(
            r"token\s+(?:expired|invalid|revoked|missing|not\s+found)", 0.88,
            description="Specific token failure phrase",
        ),
        PatternRule(
            r"authentication\s+(?:failed|required|error)", 0.85,
            description="Authentication failure phrase",
        ),
        PatternRule(
            r"invalid\s+(?:token|credentials|api.?key|api\s+key)", 0.85,
            description="Invalid credentials phrase",
        ),
        PatternRule(
            r"(?:access|permission)\s+denied", 0.82,
            description="Access/permission denied phrase",
        ),
        PatternRule(
            r"insufficient\s+(?:scope|permissions?|privileges?)", 0.80,
            description="Insufficient scope/permissions phrase",
        ),
        PatternRule(
            r"\bexpired\b", 0.35,
            description="'expired' alone — weak auth signal",
        ),
        PatternRule(
            r"\btoken\b", 0.25,
            description="'token' alone — very weak signal",
        ),
    ],

    # ── ENVIRONMENT ───────────────────────────────────────────────────────────
    FailureType.ENVIRONMENT: [
        PatternRule(
            r"connection\s*refused", 0.95,
            description="Connection refused (OS-level error)",
        ),
        PatternRule(
            r"\bconnectionrefusederror\b", 0.95,
            description="Python ConnectionRefusedError class",
        ),
        PatternRule(
            r"\beconnrefused\b", 0.93,
            description="ECONNREFUSED (Node/OS errno)",
        ),
        PatternRule(
            r"\b(502|503|504)\b", 0.92,
            description="HTTP 5xx gateway/infra error codes",
        ),
        PatternRule(
            r"5\d{2}\s+(?:bad\s+gateway|service\s+unavailable|gateway\s+timeout)", 0.90,
            description="Verbose 5xx error phrase",
        ),
        PatternRule(
            r"name\s+or\s+service\s+not\s+known", 0.92,
            description="DNS resolution failure",
        ),
        PatternRule(
            r"network\s+(?:unreachable|is\s+unreachable|error)", 0.90,
            description="Network unreachable OS error",
        ),
        PatternRule(
            r"max\s+retries?\s+exceeded", 0.85,
            description="urllib3 / requests retry exhaustion",
        ),
        PatternRule(
            r"(?:read|connect|pool)\s+timed?\s+out", 0.80,
            description="Network-level timeout phrase",
        ),
        PatternRule(
            r"\bconnectionerror\b", 0.78,
            description="Python requests.ConnectionError",
        ),
        PatternRule(
            r"\btimeouterror\b(?!.*(?:selector|locator|element))", 0.65,
            description="Generic TimeoutError (not Playwright)",
        ),
        PatternRule(
            r"\btimed?\s+out\b(?!.*(?:selector|locator|element|wait))", 0.60,
            description="'timed out' without UI selector context",
        ),
    ],

    # ── SELECTOR ──────────────────────────────────────────────────────────────
    FailureType.SELECTOR: [
        PatternRule(
            r"element\s+not\s+found", 0.95,
            description="Explicit 'element not found' message",
        ),
        PatternRule(
            r"\belementnotfound(?:exception)?\b", 0.95,
            description="ElementNotFound exception class",
        ),
        PatternRule(
            r"waiting\s+for\s+(?:selector|locator|element)", 0.93,
            description="Playwright 'waiting for selector/locator'",
        ),
        PatternRule(
            r"locator\s*\(.*\)\s*(?:timed?\s+out|not\s+visible|not\s+attached)", 0.92,
            description="Playwright locator timeout/visibility error",
        ),
        PatternRule(
            r"page\.(?:locator|get_by_\w+)\s*\(", 0.80,
            description="Playwright page.locator() call in traceback",
        ),
        PatternRule(
            r"\bplaywright\b.*(?:timeout|error|not\s+found)", 0.82,
            description="Playwright + error/timeout on same line",
        ),
        PatternRule(
            r"(?:css|xpath)\s+selector\s+not\s+found", 0.88,
            description="Explicit selector not found message",
        ),
        PatternRule(
            r"\bplaywright\b", 0.45,
            description="'playwright' alone — moderate signal",
        ),
        PatternRule(
            r"\bselector\b(?!.*schema)", 0.35,
            description="'selector' alone (not schema context) — weak signal",
        ),
    ],

    # ── SCHEMA ────────────────────────────────────────────────────────────────
    FailureType.SCHEMA: [
        PatternRule(
            r"\bvalidationerror\b", 0.95,
            description="Pydantic or marshmallow ValidationError",
        ),
        PatternRule(
            r"\bjsonschema\b", 0.93,
            description="jsonschema library reference",
        ),
        PatternRule(
            r"extra\s+fields?\s+not\s+permitted", 0.92,
            description="Pydantic v2 extra-fields error",
        ),
        PatternRule(
            r"field\s+required", 0.90,
            description="Pydantic 'field required' error",
        ),
        PatternRule(
            r"missing\s+(?:required\s+)?(?:field|key|property)", 0.90,
            description="Missing required field message",
        ),
        PatternRule(
            r"unexpected\s+(?:field|key|property)", 0.88,
            description="Unexpected field in response",
        ),
        PatternRule(
            r"additional\s+properties\s+(?:are\s+)?not\s+allowed", 0.88,
            description="JSON Schema additionalProperties violation",
        ),
        PatternRule(
            r"does\s+not\s+match\s+(?:the\s+)?schema", 0.88,
            description="Schema mismatch message",
        ),
        PatternRule(
            r"\bkeyerror\b", 0.85,
            description="Python KeyError (missing dict key → schema drift)",
        ),
        PatternRule(
            r"(?:response|body)\s+(?:schema|structure)\s+(?:changed|mismatch)", 0.85,
            description="Explicit schema change message",
        ),
        PatternRule(
            r"\bschema\b(?!.*(?:selector|playwright))", 0.30,
            description="'schema' alone (not UI context) — weak signal",
        ),
    ],

    # ── ASSERTION ─────────────────────────────────────────────────────────────
    FailureType.ASSERTION: [
        PatternRule(
            r"\bassertionerror\b", 0.90,
            description="Python AssertionError class",
        ),
        PatternRule(
            r"expected\s+.{0,80}\bgot\b", 0.88,
            description="'expected … got' pattern in test output",
        ),
        PatternRule(
            r"not\s+equal\b", 0.85,
            description="'not equal' assertion message",
        ),
        PatternRule(
            r"^E\s+assert\s+", 0.85,
            description="pytest short-repr assertion line (starts with 'E assert')",
        ),
        PatternRule(
            r"expected:\s*\S", 0.82,
            description="pytest 'Expected:' label in diff output",
        ),
        PatternRule(
            r"(?:received|actual|got):\s*\S", 0.78,
            description="pytest 'Received:' / 'Actual:' / 'Got:' label",
        ),
        PatternRule(
            r"assert\s+\w+\s*(?:==|!=|>=|<=|>|<)\s*\w+", 0.80,
            description="assert with comparison operator",
        ),
        PatternRule(
            r"!=[^=]", 0.40,
            description="'!=' inequality (alone — weak, could be schema diff)",
        ),
        PatternRule(
            r"\bassert\b", 0.30,
            description="'assert' keyword alone — weakest signal",
        ),
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Rule-based scoring
# ──────────────────────────────────────────────────────────────────────────────

def _combine_error_text(
    error_message: str,
    stderr: str,
    stdout: str,
) -> str:
    """
    Concatenate all error sources into a single string for pattern matching.
    Truncated to prevent runaway regex performance on giant outputs.
    """
    return "\n".join([
        error_message[:3000],
        stderr[:2000],
        stdout[:1000],
    ])


def _score_rules(
    text: str,
) -> Dict[FailureType, Tuple[float, List[str]]]:
    """
    Apply all rules against `text` and return per-type scores.

    Returns
    -------
    dict mapping FailureType → (accumulated_confidence, matched_descriptions)
    Confidence is capped at 1.0 per type.
    """
    scores: Dict[FailureType, Tuple[float, List[str]]] = {}

    for failure_type, rules in _RULES.items():
        accumulated = 0.0
        matched: List[str] = []

        for rule in rules:
            flags = re.IGNORECASE if rule.case_insensitive else 0
            try:
                if re.search(rule.pattern, text, flags | re.MULTILINE):
                    accumulated += rule.weight
                    matched.append(rule.description or rule.pattern)
            except re.error as exc:
                logger.warning("Bad regex pattern '%s': %s", rule.pattern, exc)

        scores[failure_type] = (min(accumulated, 1.0), matched)

    return scores


def _pick_winner(
    scores: Dict[FailureType, Tuple[float, List[str]]],
) -> Tuple[FailureType, float, List[str]]:
    """
    Select the highest-confidence type from the score table.

    Tie-breaking: the type whose rules have the higher maximum individual
    weight wins (more specific pattern = higher diagnostic value).
    """
    best_type   = FailureType.ASSERTION    # safe default
    best_score  = -1.0
    best_matches: List[str] = []

    for failure_type, (score, matches) in scores.items():
        if score > best_score:
            best_score   = score
            best_type    = failure_type
            best_matches = matches
        elif score == best_score and score > 0.0:
            # Tie: prefer the type whose highest rule weight is larger
            challenger_max = max(r.weight for r in _RULES[failure_type])
            current_max    = max(r.weight for r in _RULES[best_type])
            if challenger_max > current_max:
                best_type    = failure_type
                best_matches = matches

    return best_type, best_score, best_matches


def _rule_based_classify(
    error_message: str,
    stderr: str,
    stdout: str,
) -> ClassificationResult:
    """
    Stage 1: apply rule tables and return a ClassificationResult.
    method is always "rule_based" regardless of confidence.
    """
    text   = _combine_error_text(error_message, stderr, stdout)
    scores = _score_rules(text)
    winner, confidence, matches = _pick_winner(scores)

    # Build a short reasoning string showing what fired
    if matches:
        matched_str = "; ".join(matches[:5])  # cap at 5 for log readability
        reasoning = (
            f"Rule-based: {winner.value} (confidence={confidence:.2f}). "
            f"Matched: {matched_str}."
        )
    else:
        reasoning = (
            f"Rule-based: no patterns matched. "
            f"Defaulting to {winner.value} with confidence={confidence:.2f}."
        )

    return ClassificationResult(
        failure_type=winner,
        confidence=round(confidence, 4),
        method="rule_based",
        reasoning=reasoning,
        matched_patterns=matches,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — LLM fallback
# ──────────────────────────────────────────────────────────────────────────────

class _LLMClassifierOutput(BaseModel):
    """
    Structured output schema for the LLM failure classifier.
    with_structured_output() enforces this schema — no JSON parsing needed.
    """
    failure_type: str = Field(
        description="One of: ASSERTION, SCHEMA, SELECTOR, ENVIRONMENT, AUTH",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Classifier's self-reported confidence in [0.0, 1.0]",
    )
    reasoning: str = Field(
        description="One concise sentence citing specific evidence from the error",
    )


_VALID_FAILURE_TYPES = {ft.value.upper() for ft in FailureType if ft != FailureType.LOGIC}
_FAILURE_TYPE_MAP    = {ft.value.upper(): ft for ft in FailureType}


def _llm_classify(
    error_message: str,
    stderr: str,
    stdout: str,
) -> ClassificationResult:
    """
    Stage 2: call GPT-4o via the Vocareum proxy when rule-based confidence
    is insufficient.

    Uses with_structured_output() with _LLMClassifierOutput — no raw JSON
    parsing. If the LLM returns an unrecognised failure_type value, falls
    back to ASSERTION (the least-harmful escalation).
    """
    from langchain_openai import ChatOpenAI

    from qa_agent.healing.prompts import CLASSIFIER_SYSTEM, CLASSIFIER_USER

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=_VOCAREUM_API_KEY,
        base_url=_VOCAREUM_BASE_URL,
        temperature=0,
    ).with_structured_output(_LLMClassifierOutput)

    user_content = CLASSIFIER_USER.format(
        error_message=error_message[:2000],
        stderr=stderr[:1000],
        stdout=stdout[:500],
    )

    try:
        result: _LLMClassifierOutput = llm.invoke([
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user",   "content": user_content},
        ])

        raw_type = result.failure_type.strip().upper()

        if raw_type not in _VALID_FAILURE_TYPES:
            logger.warning(
                "LLM returned unrecognised failure_type '%s'; defaulting to ASSERTION.",
                raw_type,
            )
            failure_type = FailureType.ASSERTION
            confidence   = max(result.confidence - 0.2, 0.3)  # penalise invalid output
            reasoning    = (
                f"LLM returned invalid type '{raw_type}'; defaulted to ASSERTION. "
                f"Original reasoning: {result.reasoning}"
            )
        else:
            failure_type = _FAILURE_TYPE_MAP[raw_type]
            confidence   = result.confidence
            reasoning    = f"LLM fallback: {result.reasoning}"

    except Exception as exc:
        logger.error("LLM classification failed: %s — defaulting to ENVIRONMENT.", exc)
        # ENVIRONMENT is the safest default: it always escalates to human_review
        failure_type = FailureType.ENVIRONMENT
        confidence   = 0.40
        reasoning    = (
            f"LLM classification call failed ({type(exc).__name__}: {exc}). "
            f"Defaulted to ENVIRONMENT (non-healable) for safety."
        )

    return ClassificationResult(
        failure_type=failure_type,
        confidence=round(confidence, 4),
        method="llm_fallback",
        reasoning=reasoning,
        matched_patterns=[],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def classify_failure(
    error_message: str,
    stderr: str = "",
    stdout: str = "",
) -> ClassificationResult:
    """
    Classify a test failure into one of 5 FailureTypes using a two-stage
    rule + LLM pipeline.

    Stage 1 (rule-based) runs first.  If its confidence ≥ _LLM_THRESHOLD
    (0.6 by default), the result is returned immediately — no LLM call.

    Stage 2 (LLM fallback) runs only when Stage 1 confidence < 0.6.  It
    uses GPT-4o via the Vocareum proxy with with_structured_output() to
    avoid raw JSON parsing.

    Parameters
    ----------
    error_message : str   Primary error or exception text from the test run.
    stderr        : str   Raw stderr output from the subprocess.
    stdout        : str   Raw stdout output from the subprocess.

    Returns
    -------
    ClassificationResult
        failure_type, confidence, method ("rule_based" | "llm_fallback"),
        reasoning, and matched_patterns list.

    Notes
    -----
    - Never raises; the LLM fallback has its own exception handler that
      returns FailureType.ENVIRONMENT (non-healable, safe default).
    - The returned failure_type is always one of the 5 classifiable types.
      FailureType.LOGIC is never returned by this function.
    """
    # Stage 1
    rule_result = _rule_based_classify(error_message, stderr, stdout)

    if rule_result.confidence >= _LLM_THRESHOLD:
        logger.debug(
            "Rule-based classification: %s (conf=%.2f) — skipping LLM.",
            rule_result.failure_type.value,
            rule_result.confidence,
        )
        return rule_result

    # Stage 2
    logger.info(
        "Rule-based confidence %.2f < %.2f — invoking LLM fallback.",
        rule_result.confidence,
        _LLM_THRESHOLD,
    )
    llm_result = _llm_classify(error_message, stderr, stdout)

    # If LLM is also uncertain, prefer whichever stage returned higher confidence
    if rule_result.confidence >= llm_result.confidence and rule_result.confidence > 0.0:
        logger.debug(
            "LLM (conf=%.2f) was not more confident than rules (conf=%.2f); "
            "returning rule-based result.",
            llm_result.confidence,
            rule_result.confidence,
        )
        return rule_result

    return llm_result


def is_auto_healable(failure_type: FailureType) -> bool:
    """
    Return True if SelfHealingAgent can attempt autonomous repair for this type.

    Auto-healable types
    -------------------
    ASSERTION  — update expected value from spec
    SCHEMA     — update field access pattern from spec
    SELECTOR   — generate alternative Playwright locators

    Non-healable types (always escalated to human_review)
    ------------------------------------------------------
    ENVIRONMENT — infrastructure failure; no code change will fix it
    AUTH        — credential/permission change; requires human intervention

    Parameters
    ----------
    failure_type : FailureType

    Returns
    -------
    bool
    """
    return failure_type in _AUTO_HEALABLE


def classify_all_failures(
    test_results: list,
) -> List[ClassificationResult]:
    """
    Classify every failed/errored TestResult in a list.

    Convenience wrapper used by the self_heal node to batch-classify
    all failures from the latest ExecutionReport in one call.

    Parameters
    ----------
    test_results : list[TestResult]   All results from ExecutionReport.per_test_results.
                                      Passed as a plain list to avoid circular imports
                                      at module level; duck-typed.

    Returns
    -------
    List[ClassificationResult]
        One entry per failed/errored result, in the same order.
        Skips passed/skipped results silently.
    """
    classifications: List[ClassificationResult] = []

    for result in test_results:
        # Duck-typed: works with TestResult Pydantic model or any object
        # with .status, .error_message, .stderr, .stdout attributes.
        status = getattr(result, "status", "")
        if status not in ("failed", "error"):
            continue

        classification = classify_failure(
            error_message=getattr(result, "error_message", "") or "",
            stderr=getattr(result, "stderr", "") or "",
            stdout=getattr(result, "stdout", "") or "",
        )
        classifications.append(classification)

    return classifications
