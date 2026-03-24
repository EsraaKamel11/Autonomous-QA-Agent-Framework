"""
qa_agent/observability.py
──────────────────────────
Structured observability for the Autonomous QA Agent pipeline.

Public API
──────────
  configure_langsmith()            — enable LangSmith tracing via env vars (lazy import)
  log_node_execution(...)          — append one entry to decisions.jsonl  (thread-safe)
  log_escalation(...)              — append one entry to escalations.jsonl (thread-safe)
  track_token_cost(...)            — append one entry to token_costs.jsonl (thread-safe)
  get_run_summary(pr_number, ...)  — aggregate metrics across a pipeline run

Design
──────
- All JSONL writes are append-only and protected by per-file threading.Lock.
- Every public function is wrapped in try/except — logging failures must never
  crash the pipeline.
- LangSmith is lazily imported inside configure_langsmith() so a missing
  langsmith package does not affect the import of this module or any node.

Log files (directory controlled by QA_AGENT_LOG_DIR, default ./logs)
────────────────────────────────────────────────────────────────────────
  decisions.jsonl    — one entry per node execution (all nodes)
  escalations.jsonl  — one entry per human escalation (written before interrupt())
  token_costs.jsonl  — one entry per LLM call with token + cost estimates
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Log file paths ────────────────────────────────────────────────────────────
_LOG_DIR      = os.environ.get("QA_AGENT_LOG_DIR", "./logs")
_DECISIONS    = os.path.join(_LOG_DIR, "decisions.jsonl")
_ESCALATIONS  = os.path.join(_LOG_DIR, "escalations.jsonl")
_TOKEN_COSTS  = os.path.join(_LOG_DIR, "token_costs.jsonl")

# ── Per-file write locks (prevents interleaved writes from parallel branches) ──
_locks_mutex: threading.Lock = threading.Lock()
_file_locks:  Dict[str, threading.Lock] = {}


def _get_lock(path: str) -> threading.Lock:
    """Return (creating on demand) a per-file threading.Lock."""
    with _locks_mutex:
        if path not in _file_locks:
            _file_locks[path] = threading.Lock()
        return _file_locks[path]


def _append_jsonl(path: str, record: dict) -> None:
    """
    Atomically append one JSON line to *path*.

    Creates parent directories on first write.  Acquires the per-file lock
    before opening so concurrent writes from parallel graph branches do not
    interleave partial lines.  Never raises.
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        with _get_lock(path):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception as exc:
        logger.debug("observability._append_jsonl failed for %s: %s", path, exc)


def _truncate(value: Any, max_chars: int = 300) -> str:
    """Safely stringify and truncate any value for log compactness."""
    try:
        text = str(value)
    except Exception:
        text = "<unstringifiable>"
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


# ──────────────────────────────────────────────────────────────────────────────
# 1. LangSmith tracing
# ──────────────────────────────────────────────────────────────────────────────

def configure_langsmith(project_name: str = "qa-agent") -> bool:
    """
    Enable LangSmith tracing for the current process.

    Reads LANGCHAIN_API_KEY from the environment.  Returns False (never raises)
    when the key is absent or the langsmith package is not installed.

    Sets LANGCHAIN_TRACING_V2=true and LANGCHAIN_PROJECT=<project_name> so
    all LangChain/LangGraph calls in this process are automatically traced.

    Parameters
    ----------
    project_name : str  LangSmith project name. Default: "qa-agent".

    Returns
    -------
    bool  True if tracing was successfully enabled.
    """
    try:
        api_key = os.environ.get("LANGCHAIN_API_KEY", "")
        if not api_key:
            logger.debug(
                "configure_langsmith: LANGCHAIN_API_KEY not set — tracing disabled"
            )
            return False

        # Lazy import — never crash if langsmith package is absent
        import langsmith  # noqa: F401  # type: ignore[import]

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"]    = project_name
        os.environ.setdefault(
            "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
        )

        logger.info("LangSmith tracing enabled — project: %s", project_name)
        return True

    except ImportError:
        logger.debug(
            "configure_langsmith: langsmith package not installed — tracing disabled"
        )
        return False
    except Exception as exc:
        logger.debug("configure_langsmith failed: %s", exc)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 2. Node execution log
# ──────────────────────────────────────────────────────────────────────────────

def log_node_execution(
    node_name:      str,
    input_summary:  Dict[str, Any],
    output_summary: Dict[str, Any],
    routing_hint:   str           = "",
    pr_number:      int           = 0,
    duration_ms:    float         = 0.0,
    error:          Optional[str] = None,
) -> None:
    """
    Append one structured entry to ./logs/decisions.jsonl.

    Called by every node immediately before it returns its state dict.
    Never raises — a logging failure must never crash the pipeline.

    Record schema
    -------------
    {
      "timestamp":      ISO-8601 UTC string,
      "node":           node_name,
      "pr_number":      int,
      "duration_ms":    float,
      "input_summary":  {key: truncated_str, ...},
      "output_summary": {key: truncated_str, ...},
      "routing_hint":   str,
      "error":          str | null
    }
    """
    try:
        record: dict = {
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "node":           node_name,
            "pr_number":      pr_number,
            "duration_ms":    round(duration_ms, 2),
            "input_summary":  {k: _truncate(v) for k, v in input_summary.items()},
            "output_summary": {k: _truncate(v) for k, v in output_summary.items()},
            "routing_hint":   routing_hint,
            "error":          error,
        }
        _append_jsonl(_DECISIONS, record)
    except Exception as exc:
        logger.debug("log_node_execution failed silently: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Escalation log
# ──────────────────────────────────────────────────────────────────────────────

def log_escalation(
    escalation_reason: str,
    pr_number:         Optional[int]           = None,
    repo:              Optional[str]           = None,
    context:           Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append one entry to ./logs/escalations.jsonl.

    This is the append-only audit trail for every HITL escalation event.
    Must be written BEFORE interrupt() is called in the human_review node
    so the record exists even if the process is killed before resumption.

    Parameters
    ----------
    escalation_reason : str   Short description of why human review is required.
    pr_number         : int   GitHub PR number (for log correlation).
    repo              : str   GitHub repository slug (owner/repo).
    context           : dict  Additional fields (critical defects, score, etc.).
    """
    try:
        record: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type":      "human_escalation",
            "pr_number": pr_number,
            "repo":      repo,
            "reason":    escalation_reason,
            "context":   context or {},
        }
        _append_jsonl(_ESCALATIONS, record)
    except Exception as exc:
        logger.debug("log_escalation failed silently: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Token cost tracking
# ──────────────────────────────────────────────────────────────────────────────

# Approximate cost per 1 000 tokens (USD) — update when pricing changes.
_COST_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-4o":                 {"prompt": 0.0025,  "completion": 0.0100},
    "gpt-4o-mini":            {"prompt": 0.00015, "completion": 0.00060},
    "gpt-4-turbo":            {"prompt": 0.0100,  "completion": 0.0300},
    "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.0},
    "text-embedding-ada-002": {"prompt": 0.00010, "completion": 0.0},
}
_DEFAULT_COST_RATES = {"prompt": 0.0025, "completion": 0.0100}


def _estimate_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Return estimated USD cost for the given model and token counts."""
    rates = _COST_PER_1K.get(model, _DEFAULT_COST_RATES)
    return (
        prompt_tokens    / 1_000.0 * rates["prompt"]
        + completion_tokens / 1_000.0 * rates["completion"]
    )


def track_token_cost(
    agent_name:     str,
    usage_response: Any,
    model:          str           = "gpt-4o",
    pr_number:      Optional[int] = None,
    run_id:         Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extract token counts from an OpenAI usage object and log to token_costs.jsonl.

    Accepted input types for *usage_response*:
    • OpenAI CompletionUsage object  (.prompt_tokens / .completion_tokens)
    • LangChain AIMessage            (.usage_metadata with input_tokens/output_tokens)
    • Plain dict                     ("prompt_tokens" / "completion_tokens" keys)

    Returns the cost record dict, or None if extraction failed.  Never raises.
    """
    try:
        prompt_tokens:     int = 0
        completion_tokens: int = 0

        if hasattr(usage_response, "prompt_tokens"):
            # OpenAI CompletionUsage
            prompt_tokens     = int(usage_response.prompt_tokens     or 0)
            completion_tokens = int(usage_response.completion_tokens or 0)

        elif hasattr(usage_response, "usage_metadata"):
            # LangChain AIMessage
            meta              = usage_response.usage_metadata or {}
            prompt_tokens     = int(meta.get("input_tokens",  0))
            completion_tokens = int(meta.get("output_tokens", 0))

        elif isinstance(usage_response, dict):
            prompt_tokens     = int(usage_response.get("prompt_tokens",     0))
            completion_tokens = int(usage_response.get("completion_tokens", 0))

        total_tokens = prompt_tokens + completion_tokens
        cost_usd     = _estimate_cost_usd(model, prompt_tokens, completion_tokens)

        record: dict = {
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "agent":            agent_name,
            "model":            model,
            "pr_number":        pr_number,
            "run_id":           run_id,
            "prompt_tokens":    prompt_tokens,
            "completion_tokens":completion_tokens,
            "total_tokens":     total_tokens,
            "cost_usd":         round(cost_usd, 6),
        }
        _append_jsonl(_TOKEN_COSTS, record)
        return record

    except Exception as exc:
        logger.debug("track_token_cost failed for %s: %s", agent_name, exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 5. Run summary
# ──────────────────────────────────────────────────────────────────────────────

def get_run_summary(
    pr_number: Optional[int] = None,
    run_id:    Optional[str] = None,
) -> Dict[str, Any]:
    """
    Read the three JSONL log files and return aggregated metrics for one run.

    Filters by *pr_number* when provided.  Filters token_costs.jsonl by
    *run_id* when provided (in addition to pr_number filtering).

    Returns
    -------
    dict with keys:
      nodes_executed   int    — number of node execution log entries
      total_tokens     int    — cumulative tokens across all LLM calls
      cost_usd         float  — estimated total USD spend
      final_verdict    str    — last merge_recommendation seen in output_summary
      final_score      float  — last overall_score seen in output_summary
      has_errors       bool   — any node logged a non-null error
      escalations      int    — number of human escalation events
      duration_ms_sum  float  — sum of all node duration_ms values
    """
    summary: Dict[str, Any] = {
        "nodes_executed":  0,
        "total_tokens":    0,
        "cost_usd":        0.0,
        "final_verdict":   None,
        "final_score":     None,
        "has_errors":      False,
        "escalations":     0,
        "duration_ms_sum": 0.0,
    }

    try:
        # ── Decision log ─────────────────────────────────────────────────────
        if Path(_DECISIONS).exists():
            with open(_DECISIONS, encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if pr_number is not None and entry.get("pr_number") != pr_number:
                        continue

                    summary["nodes_executed"] += 1

                    dur = entry.get("duration_ms")
                    if dur is not None:
                        summary["duration_ms_sum"] += float(dur)

                    if entry.get("error"):
                        summary["has_errors"] = True

                    out = entry.get("output_summary", {})
                    if isinstance(out, dict):
                        if "verdict" in out:
                            summary["final_verdict"] = out["verdict"]
                        if "overall_score" in out and out["overall_score"] is not None:
                            summary["final_score"] = out["overall_score"]

        # ── Token cost log ────────────────────────────────────────────────────
        if Path(_TOKEN_COSTS).exists():
            with open(_TOKEN_COSTS, encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if pr_number is not None and entry.get("pr_number") != pr_number:
                        continue
                    if run_id is not None and entry.get("run_id") != run_id:
                        continue

                    summary["total_tokens"] += int(entry.get("total_tokens", 0))
                    summary["cost_usd"]     += float(entry.get("cost_usd",   0.0))

        # ── Escalation log ────────────────────────────────────────────────────
        if Path(_ESCALATIONS).exists():
            with open(_ESCALATIONS, encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if pr_number is not None and entry.get("pr_number") != pr_number:
                        continue
                    summary["escalations"] += 1

        summary["cost_usd"] = round(summary["cost_usd"], 6)

    except Exception as exc:
        logger.debug("get_run_summary failed: %s", exc)

    return summary
