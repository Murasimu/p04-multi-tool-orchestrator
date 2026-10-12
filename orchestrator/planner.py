"""Planner: turn a natural-language task into a validated, ordered plan.

The LLM proposes JSON: a list of steps (tool, args, optional depends_on).
The planner validates every step against the registry (tool exists,
dependencies acyclic), so bad LLM plans are rejected structurally rather
than exploding mid-execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .registry import ToolRegistry

LLM = Callable[[list[dict[str, str]]], str]

PLANNER_PROMPT = """You are a workflow planner. Given a task and a tool catalog,
produce a plan as a JSON array. Each element:
{{"tool": "<name>", "args": {{...}}, "depends_on": [<indices or names>]}}

- Use ONLY tools from the catalog.
- Steps with no dependencies can run in parallel.
- Keep plans minimal: no more steps than needed.

Tool catalog:
{catalog}

Task: {task}

Respond with ONLY the JSON array."""


class PlanError(ValueError):
    pass


@dataclass
class PlanStep:
    name: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)
    raw: str = ""


@dataclass
class Planner:
    registry: ToolRegistry
    llm: LLM

    def plan(self, task: str, max_steps: int = 10) -> Plan:
        """Ask the LLM for a plan and validate it structurally."""
        raw = self.llm([{"role": "user", "content": PLANNER_PROMPT.format(
            catalog=self.registry.describe(), task=task)}])

        steps_data = _extract_array(raw)
        if not isinstance(steps_data, list) or not steps_data:
            raise PlanError("plan must be a non-empty JSON array")
        if len(steps_data) > max_steps:
            raise PlanError(f"plan exceeds max_steps={max_steps}")

        plan = Plan(raw=raw)
        step_names: set[str] = set()
        for i, item in enumerate(steps_data):
            if not isinstance(item, dict) or "tool" not in item:
                raise PlanError(f"step {i} missing 'tool'")
            name = item.get("name") or f"step-{i}"
            tool = item["tool"]
            if tool not in self.registry.tools:
                raise PlanError(f"step {i} references unknown tool {tool!r}")
            deps = item.get("depends_on", [])
            if not isinstance(deps, list):
                raise PlanError(f"step {i} depends_on must be a list")
            # LLMs often use positional indices OR tool names: resolve both.
            resolved: list[str] = []
            for d in deps:
                if isinstance(d, int) or (isinstance(d, str) and d.isdigit()):
                    idx = int(d)
                    if not (0 <= idx < len(steps_data)):
                        raise PlanError(f"step {i} depends on out-of-range index {idx}")
                    dep_name = steps_data[idx].get("name") or f"step-{idx}"
                    if dep_name == str(name):
                        raise PlanError(f"step {i} depends on itself")
                    resolved.append(str(dep_name))
                elif isinstance(d, str) and d in self.registry.tools and d not in step_names:
                    # a tool name used as a dependency -> first step running it
                    matches = [s for s in plan.steps if s.tool == d]
                    if not matches:
                        raise PlanError(
                            f"step {i} depends on tool {d!r} but no earlier step uses it")
                    resolved.append(matches[0].name)
                else:
                    resolved.append(str(d))
            plan.steps.append(PlanStep(
                name=str(name), tool=tool,
                args=dict(item.get("args") or {}),
                depends_on=resolved,
            ))
            step_names.add(str(name))

        self._check_acyclic(plan)
        return plan

    def _check_acyclic(self, plan: Plan) -> None:
        by_name = {s.name: s for s in plan.steps}
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(step: PlanStep) -> None:
            if step.name in done:
                return
            if step.name in visiting:
                raise PlanError(f"dependency cycle at {step.name!r}")
            visiting.add(step.name)
            for dep in step.depends_on:
                if dep in by_name:
                    visit(by_name[dep])
                else:
                    raise PlanError(f"{step.name!r} depends on unknown step {dep!r}")
            visiting.discard(step.name)
            done.add(step.name)

        for step in plan.steps:
            visit(step)


def _extract_array(raw: str) -> list | None:
    """Pull the first JSON array out of planner output."""
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
