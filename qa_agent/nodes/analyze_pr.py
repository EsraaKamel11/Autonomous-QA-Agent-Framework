"""
qa_agent/nodes/analyze_pr.py
──────────────────────────────
CodeAnalystAgent — node 1 of 12.

Decision boundary
─────────────────
Only answers: "What changed in this pull request and how risky is it?"
Never generates tests, reads specs, or makes routing decisions.

Inputs consumed from state
──────────────────────────
  state["pr_metadata"]   PRMetadata — pr_number, repo, head_sha, author, title

State keys written
──────────────────
  change_manifest   ChangeManifest
  current_phase     str
"""

from __future__ import annotations

import json
import os
import time
from typing import List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from qa_agent.observability import log_node_execution
from qa_agent.prompts.code_analyst import CODE_ANALYST_SYSTEM, CODE_ANALYST_USER
from qa_agent.state import (
    AffectedEndpoint,
    AgentError,
    ChangeManifest,
    QAAgentState,
)
from qa_agent.tools.github_tools import ChangedFile, get_changed_files, get_pr_diff

# ──────────────────────────────────────────────────────────────────────────────
# Vocareum proxy credentials — always passed explicitly, never from env default
# ──────────────────────────────────────────────────────────────────────────────

_VOCAREUM_API_KEY  = os.environ.get("VOCAREUM_API_KEY",  "voc-placeholder")
_VOCAREUM_BASE_URL = os.environ.get("VOCAREUM_BASE_URL", "https://openai.vocareum.com/v1")
_NODE_NAME = "analyze_pr"


# ──────────────────────────────────────────────────────────────────────────────
# Intermediate LLM output model
#
# We cannot use ChangeManifest directly with with_structured_output because
# AffectedEndpoint is a nested model.  We define a flat LLM-facing model
# and convert it to the canonical state type after.
# ──────────────────────────────────────────────────────────────────────────────

class _LLMAffectedEndpoint(BaseModel):
    method:       str
    path:         str
    change_type:  str = "modified"   # added | modified | removed
    operation_id: str = ""


class _LLMChangeManifest(BaseModel):
    affected_files:     List[str]
    affected_endpoints: List[_LLMAffectedEndpoint]
    affected_modules:   List[str]
    risk_score:         float = Field(ge=0.0, le=1.0)
    change_summary:     str


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _format_changed_files(files: List[ChangedFile]) -> str:
    lines = []
    for f in files:
        lines.append(
            f"{f.status.upper():<10} | {f.filename:<60} | "
            f"+{f.additions} -{f.deletions}"
        )
    return "\n".join(lines) if lines else "(no changed files)"


def _diff_truncated(diff: str, max_chars: int = 12_000) -> str:
    """Truncate diff to stay within context limits; append ellipsis note."""
    if len(diff) <= max_chars:
        return diff
    return (
        diff[:max_chars]
        + f"\n\n... [TRUNCATED — diff exceeds {max_chars} chars] ..."
    )


def _llm_manifest_to_state(llm: _LLMChangeManifest) -> ChangeManifest:
    """Convert flat LLM output model → canonical ChangeManifest."""
    endpoints = [
        AffectedEndpoint(
            method=ep.method.upper(),
            path=ep.path,
            change_type=ep.change_type,
            operation_id=ep.operation_id or None,
        )
        for ep in llm.affected_endpoints
    ]
    return ChangeManifest(
        affected_files=llm.affected_files,
        affected_endpoints=endpoints,
        affected_modules=llm.affected_modules,
        risk_score=round(llm.risk_score, 4),
        change_summary=llm.change_summary,
    )


def _fallback_manifest(pr_metadata, error: str) -> ChangeManifest:
    """Safe fallback when LLM call fails — preserves pipeline continuity."""
    return ChangeManifest(
        affected_files=[],
        affected_endpoints=[],
        affected_modules=[],
        risk_score=0.5,   # conservative mid-range: not trivial, not critical
        change_summary=(
            f"[FALLBACK] CodeAnalystAgent failed: {error[:200]}. "
            f"PR #{pr_metadata.pr_number} in {pr_metadata.repo}. "
            f"Manual review required."
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────────────────────────────────────

def analyze_pr(state: QAAgentState) -> dict:
    """
    CodeAnalystAgent: fetch the PR diff and extract a ChangeManifest.

    Pipeline position: Node 1 — first node after START.
    Input:  state["pr_metadata"]
    Output: state["change_manifest"]

    LLM call: yes — with_structured_output(_LLMChangeManifest) via Vocareum.
    Tool calls: github_tools.get_pr_diff, github_tools.get_changed_files.
    """
    start_ts    = time.time()
    pr_metadata = state["pr_metadata"]
    errors      = []

    # ── Step 1: Fetch raw diff and file list ──────────────────────────────────
    diff          = get_pr_diff(pr_metadata.pr_number, pr_metadata.repo)
    changed_files = get_changed_files(pr_metadata.pr_number, pr_metadata.repo)

    # ── Step 2: Format context for LLM ───────────────────────────────────────
    files_text = _format_changed_files(changed_files)
    diff_text  = _diff_truncated(diff)

    user_message = CODE_ANALYST_USER.format(
        pr_title=pr_metadata.title,
        pr_description=pr_metadata.description[:2000] or "(no description)",
        repo=pr_metadata.repo,
        changed_files=files_text,
        diff=diff_text,
    )

    # ── Step 3: LLM call with structured output ───────────────────────────────
    error_str: str | None = None
    manifest: ChangeManifest

    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=_VOCAREUM_API_KEY,
            base_url=_VOCAREUM_BASE_URL,
            temperature=0,
        ).with_structured_output(_LLMChangeManifest)

        llm_result: _LLMChangeManifest = llm.invoke([
            {"role": "system", "content": CODE_ANALYST_SYSTEM},
            {"role": "user",   "content": user_message},
        ])
        manifest = _llm_manifest_to_state(llm_result)

    except Exception as exc:
        error_str = f"{type(exc).__name__}: {exc}"
        errors.append(AgentError(
            agent="CodeAnalystAgent",
            phase="analyze_pr",
            error=error_str,
            recoverable=True,
        ))
        manifest = _fallback_manifest(pr_metadata, error_str)

    # ── Step 4: Log before returning ─────────────────────────────────────────
    duration_ms = (time.time() - start_ts) * 1000
    log_node_execution(
        node_name=_NODE_NAME,
        input_summary={
            "pr_number":     pr_metadata.pr_number,
            "repo":          pr_metadata.repo,
            "changed_files": len(changed_files),
            "diff_chars":    len(diff),
        },
        output_summary={
            "affected_endpoints": len(manifest.affected_endpoints),
            "affected_files":     len(manifest.affected_files),
            "risk_score":         manifest.risk_score,
            "change_summary":     manifest.change_summary[:120],
        },
        routing_hint="parse_specs",
        pr_number=pr_metadata.pr_number,
        duration_ms=duration_ms,
        error=error_str,
    )

    # ── Step 5: Return only the keys this node writes ─────────────────────────
    result: dict = {
        "change_manifest": manifest,
        "current_phase":   _NODE_NAME,
    }
    if errors:
        result["errors"] = errors   # operator.add — accumulated
    return result
