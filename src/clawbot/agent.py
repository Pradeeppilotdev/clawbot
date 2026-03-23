from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .llm import LLMClient, safe_json_loads
from .memory import RunMemory
from .tools import ToolRegistry


PLANNER_PROMPT = """You are Clawbot Planner.
Create a short execution plan for the user's task.
Return strict JSON only with this schema:
{
    "intent": "one sentence",
    "research_needed": true,
    "research_queries": ["query1", "query2"],
    "output_shape": ["section name", "section name"]
}

Rules:
- Keep research_queries <= 3 and high-signal.
- If task can be solved from reasoning alone, set research_needed=false.
"""

WRITER_PROMPT = """You are Clawbot Writer.
Produce a sharp, practical answer for the user.
Use this style:
- concise but high signal
- concrete steps and trade-offs
- no fluff

If sources are provided, cite them in markdown as [n](url).
"""

CRITIC_PROMPT = """You are Clawbot Reviewer.
Return strict JSON only:
{"pass": true|false, "feedback": "...", "missing": ["..."]}
Set pass=false when answer is vague, not actionable, or misses the user's ask.
"""


@dataclass
class AgentResult:
    answer: str
    steps_used: int
    trace: list[dict[str, Any]]
    elapsed_ms: int
    mode: str


class ClawbotAgent:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        tools: ToolRegistry,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._tools = tools

    def run(self, task: str) -> AgentResult:
        started = time.perf_counter()
        memory = RunMemory(task=task)
        mode = self._settings.agent_mode.strip().lower() or "fast"
        source_urls: list[str] = []
        evidence_chunks: list[str] = []

        # Stage 1: plan once.
        plan = self._plan(task)
        memory.add("plan", {"plan": plan})

        # Stage 2: bounded research.
        step_count = 1
        local_context = self._collect_local_context(task, memory)
        if local_context:
            evidence_chunks.extend(local_context)
            step_count += len(local_context)

        if self._settings.enable_web_research and bool(plan.get("research_needed", True)):
            queries = plan.get("research_queries", [])
            if not isinstance(queries, list):
                queries = []

            budget = self._effective_research_budget(mode)
            for query in [str(q).strip() for q in queries if str(q).strip()][:budget]:
                tool_result = self._tools.call("web_search", query)
                source_urls = self._merge_sources(source_urls, self._extract_sources("web_search", tool_result))
                evidence_chunks.append(tool_result)
                step_count += 1
                memory.add("tool_result", {"tool": "web_search", "input": query, "result": tool_result})

        # Stage 3: write final answer.
        final_answer = self._write_answer(task, plan, evidence_chunks, source_urls)
        memory.add("draft", {"answer": final_answer})

        # Optional reviewer pass for non-fast modes.
        if self._should_review(mode):
            critique = self._critique(task, final_answer)
            memory.add("critique", {"critique": critique})
            if not critique.get("pass"):
                final_answer = self._revise_answer(task, final_answer, critique, evidence_chunks, source_urls)
                memory.add("revision", {"answer": final_answer})
                step_count += 1

        final_answer = self._ensure_citations(final_answer, source_urls)
        if not final_answer.strip():
            final_answer = self._heuristic_fallback_answer(task)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        memory.add("timing", {"elapsed_ms": elapsed_ms, "mode": mode})
        return AgentResult(
            answer=final_answer,
            steps_used=step_count,
            trace=memory.events,
            elapsed_ms=elapsed_ms,
            mode=mode,
        )

    def _collect_local_context(self, task: str, memory: RunMemory) -> list[str]:
        pattern = re.compile(r'([A-Za-z]:\\[^\s"\']+|[^\s"\']+\.(?:csv|txt|md|json))', re.IGNORECASE)
        matches = [m.group(1) for m in pattern.finditer(task)]
        unique_paths: list[str] = []
        for p in matches:
            if p not in unique_paths:
                unique_paths.append(p)

        contexts: list[str] = []
        for p in unique_paths[:2]:
            low = p.lower()
            if low.endswith(".csv"):
                result = self._tools.call("csv_profile", p)
                contexts.append(result)
                memory.add("tool_result", {"tool": "csv_profile", "input": p, "result": result})
            elif low.endswith((".txt", ".md", ".json")):
                result = self._tools.call("read_text_file", p)
                contexts.append(result)
                memory.add("tool_result", {"tool": "read_text_file", "input": p, "result": result})

        return contexts

    def _plan(self, task: str) -> dict[str, Any]:
        prompt = (
            f"Task:\n{task}\n\n"
            "Return valid JSON only."
        )
        raw = self._llm.complete(PLANNER_PROMPT, prompt)
        plan = safe_json_loads(raw)
        if "intent" not in plan:
            return {
                "intent": "Deliver a practical answer.",
                "research_needed": False,
                "research_queries": [],
                "output_shape": ["Summary", "Action Plan", "Risks"],
            }
        return plan

    def _write_answer(
        self,
        task: str,
        plan: dict[str, Any],
        evidence_chunks: list[str],
        source_urls: list[str],
    ) -> str:
        prompt = (
            f"Task:\n{task}\n\n"
            f"Plan JSON:\n{json.dumps(plan, ensure_ascii=True)}\n\n"
            f"Evidence JSON list:\n{json.dumps(evidence_chunks, ensure_ascii=True)}\n\n"
            f"Source URLs:\n{json.dumps(source_urls, ensure_ascii=True)}\n\n"
            "Write the final response now."
        )
        raw = self._llm.complete(WRITER_PROMPT, prompt)
        parsed = safe_json_loads(raw)
        if parsed.get("type") == "final":
            content = str(parsed.get("content", "")).strip()
            if self._looks_like_provider_outage(content):
                return self._deterministic_strategy_answer(task, plan, evidence_chunks)
            return content
        if self._looks_like_provider_outage(raw):
            return self._deterministic_strategy_answer(task, plan, evidence_chunks)
        return raw.strip()

    def _revise_answer(
        self,
        task: str,
        draft: str,
        critique: dict[str, Any],
        evidence_chunks: list[str],
        source_urls: list[str],
    ) -> str:
        prompt = (
            f"Task:\n{task}\n\n"
            f"Current draft:\n{draft}\n\n"
            f"Critique JSON:\n{json.dumps(critique, ensure_ascii=True)}\n\n"
            f"Evidence JSON list:\n{json.dumps(evidence_chunks, ensure_ascii=True)}\n\n"
            f"Source URLs:\n{json.dumps(source_urls, ensure_ascii=True)}\n\n"
            "Return improved final response only."
        )
        raw = self._llm.complete(WRITER_PROMPT, prompt)
        parsed = safe_json_loads(raw)
        if parsed.get("type") == "final":
            content = str(parsed.get("content", "")).strip()
            if self._looks_like_provider_outage(content):
                return self._deterministic_strategy_answer(task, {}, evidence_chunks)
            return content
        if self._looks_like_provider_outage(raw):
            return self._deterministic_strategy_answer(task, {}, evidence_chunks)
        return raw.strip() or draft

    def _effective_research_budget(self, mode: str) -> int:
        base = max(0, self._settings.research_query_budget)
        if mode == "fast":
            return min(base, 1)
        if mode == "deep":
            return min(max(base, 2), 4)
        return min(max(base, 1), 3)

    def _should_review(self, mode: str) -> bool:
        if self._settings.enable_critic:
            return True
        return mode == "deep"

    def _critique(self, task: str, answer: str) -> dict[str, Any]:
        prompt = (
            f"Task:\n{task}\n\n"
            f"Candidate answer:\n{answer}\n\n"
            "Return valid JSON only."
        )
        raw = self._llm.complete(CRITIC_PROMPT, prompt)
        parsed = safe_json_loads(raw)

        if "pass" not in parsed:
            # If critic fails, do a minimal heuristic fallback.
            parsed = {
                "pass": len(answer) > 80,
                "feedback": "Critic fallback applied due to invalid critic JSON.",
            }
        return parsed

    def _extract_sources(self, tool_name: str, tool_result: str) -> list[str]:
        urls: list[str] = []
        if tool_name not in {"web_search", "fetch_url"}:
            return urls

        try:
            parsed = json.loads(tool_result)
        except json.JSONDecodeError:
            return urls

        if isinstance(parsed, dict):
            single_url = parsed.get("url")
            if isinstance(single_url, str) and single_url.startswith(("http://", "https://")):
                urls.append(single_url)

            results = parsed.get("results")
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        item_url = item.get("url")
                        if isinstance(item_url, str) and item_url.startswith(("http://", "https://")):
                            urls.append(item_url)

        return urls

    def _merge_sources(self, existing: list[str], incoming: list[str]) -> list[str]:
        merged = list(existing)
        for url in incoming:
            if url not in merged:
                merged.append(url)
        return merged

    def _ensure_citations(self, answer: str, source_urls: list[str]) -> str:
        if not source_urls:
            return answer

        has_citation = re.search(r"\[\d+\]\(https?://[^)]+\)", answer) is not None
        if has_citation:
            return answer

        lines = [answer.strip(), "", "Sources"]
        for idx, url in enumerate(source_urls[:5], start=1):
            lines.append(f"{idx}. [{idx}]({url})")
        return "\n".join(lines)

    def _heuristic_fallback_answer(self, task: str) -> str:
        if "hackathon" in task.lower():
            return (
                "Winning strategy: pick one painful workflow, build a narrow but complete agent that saves at least "
                "30-50 percent of user time, and prove reliability with a short benchmark. "
                "Deliverables: a live demo, measurable before/after metric, failure handling, and a clear business case. "
                "Execution: plan in 2-hour blocks, prioritize core loop + one killer tool integration first, then polish demo narrative."
            )
        return (
            "I can deliver this quickly with a strong agent workflow: clarify objective, propose strategy options, "
            "choose a plan with trade-offs, and provide an execution checklist with risks and validation metrics."
        )

    def _looks_like_provider_outage(self, text: str) -> bool:
        lower = text.lower()
        return (
            "all configured llm providers are temporarily unavailable" in lower
            or "llm_provider_error" in lower
            or "http 429" in lower
            or "http 404" in lower
        )

    def _deterministic_strategy_answer(
        self,
        task: str,
        plan: dict[str, Any],
        evidence_chunks: list[str],
    ) -> str:
        intent = str(plan.get("intent", "Solve the task with practical execution.")).strip()
        shape = plan.get("output_shape", ["Strategy", "Execution", "Risks", "Metrics"])
        if not isinstance(shape, list) or not shape:
            shape = ["Strategy", "Execution", "Risks", "Metrics"]

        hints: list[str] = []
        csv_profiles: list[dict[str, Any]] = []
        notes_excerpts: list[str] = []
        for chunk in evidence_chunks[:1]:
            try:
                parsed = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                if "profile" in parsed and "columns" in parsed:
                    csv_profiles.append(parsed)
                if "excerpt" in parsed and "path" in parsed:
                    notes_excerpts.append(str(parsed.get("excerpt", "")))
                results = parsed.get("results", [])
                if isinstance(results, list):
                    for item in results[:3]:
                        if isinstance(item, dict):
                            title = str(item.get("title", "")).strip()
                            if title:
                                hints.append(title)

        for chunk in evidence_chunks[1:]:
            try:
                parsed = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                if "profile" in parsed and "columns" in parsed:
                    csv_profiles.append(parsed)
                if "excerpt" in parsed and "path" in parsed:
                    notes_excerpts.append(str(parsed.get("excerpt", "")))

        data_specific = self._build_data_specific_fallback(task, csv_profiles, notes_excerpts)
        if data_specific:
            return data_specific

        answer_lines = [
            f"Objective: {intent}",
            "",
            f"Task: {task}",
            "",
            f"1. {shape[0]}",
            "Focus on a narrow high-impact workflow, define one measurable success metric, and ship a complete path from input to verified output.",
            "",
            f"2. {shape[1] if len(shape) > 1 else 'Execution'}",
            "Build in this order: planner, tool execution layer, output verifier, and demo script. Keep each step observable with logs and checkpoints.",
            "",
            f"3. {shape[2] if len(shape) > 2 else 'Risks'}",
            "Main risks are provider outages, tool noise, and vague prompts. Mitigate with fallback providers, source filtering, and strict output schemas.",
            "",
            f"4. {shape[3] if len(shape) > 3 else 'Metrics'}",
            "Track latency, task success rate, and recovery rate after tool/provider failures. Improve one metric each iteration.",
        ]

        if hints:
            answer_lines.extend([
                "",
                "Context hints from current retrieval:",
                "- " + "\n- ".join(hints[:3]),
            ])

        answer_lines.extend(
            [
                "",
                "Note: external model providers are currently rate-limited, so this response used local deterministic fallback logic.",
            ]
        )
        return "\n".join(answer_lines)

    def _build_data_specific_fallback(
        self,
        task: str,
        csv_profiles: list[dict[str, Any]],
        notes_excerpts: list[str],
    ) -> str:
        if not csv_profiles and not notes_excerpts:
            return ""

        task_lower = task.lower()
        wants_plan = "plan" in task_lower or "7-day" in task_lower or "7 day" in task_lower

        best_findings: list[str] = []
        metric_targets: list[str] = []

        for profile in csv_profiles:
            columns = [str(c).lower() for c in profile.get("columns", [])]
            p = profile.get("profile", {})
            if not isinstance(p, dict):
                continue

            conv_key = next((k for k in p.keys() if "conversion" in k.lower() or "rate" in k.lower()), None)
            leads_key = next((k for k in p.keys() if "lead" in k.lower() or "traffic" in k.lower()), None)
            revenue_key = next((k for k in p.keys() if "revenue" in k.lower() or "sales" in k.lower()), None)

            if conv_key and isinstance(p.get(conv_key), dict):
                conv = p[conv_key]
                cmin = conv.get("min")
                cmax = conv.get("max")
                cavg = conv.get("avg")
                if cmin is not None and cmax is not None and cavg is not None:
                    best_findings.append(
                        f"Conversion spread is wide ({cmin:.2f} to {cmax:.2f}, avg {cavg:.2f}), so conversion uplift is the highest-leverage lever."
                    )
                    target = min(float(cmax), float(cavg) + 0.02)
                    metric_targets.append(f"Raise average conversion_rate from {float(cavg):.2f} to {target:.2f} in 7 days.")

            if leads_key and isinstance(p.get(leads_key), dict):
                leads = p[leads_key]
                lmax = leads.get("max")
                lavg = leads.get("avg")
                if lmax is not None and lavg is not None:
                    best_findings.append(
                        f"Top-of-funnel volume is strong (max {float(lmax):.0f}, avg {float(lavg):.0f}); focus should be post-lead conversion, not lead volume." 
                    )

            if revenue_key and isinstance(p.get(revenue_key), dict):
                rev = p[revenue_key]
                ravg = rev.get("avg")
                if ravg is not None:
                    metric_targets.append(f"Target +10% weekly revenue lift from baseline avg {float(ravg):.0f}.")

            if columns:
                best_findings.append(f"Detected columns: {', '.join(columns)}.")

        note_blob = "\n".join(notes_excerpts).lower()
        if "weak conversion" in note_blob:
            best_findings.append("Notes confirm conversion bottleneck in at least one segment/team.")
        if "messaging" in note_blob:
            best_findings.append("Notes suggest messaging quality correlates with better close rates.")

        if not best_findings:
            return ""

        lines = ["Data-backed takeaway", ""]
        lines.extend([f"- {x}" for x in best_findings[:4]])

        if wants_plan:
            lines.extend(
                [
                    "",
                    "7-day action plan",
                    "1. Day 1: Segment by source/team and validate baseline conversion by segment.",
                    "2. Day 2-3: Rewrite top messaging touchpoints and deploy A/B variants.",
                    "3. Day 4-5: Route high-intent leads to the best-performing script/workflow.",
                    "4. Day 6: Remove low-performing variant and standardize winner.",
                    "5. Day 7: Review KPI delta and lock next-week iteration targets.",
                ]
            )

        if metric_targets:
            lines.extend(["", "Metric targets"])
            lines.extend([f"- {x}" for x in metric_targets[:3]])

        lines.extend(
            [
                "",
                "Note: external model providers are currently rate-limited, so this response used local deterministic fallback logic.",
            ]
        )
        return "\n".join(lines)
