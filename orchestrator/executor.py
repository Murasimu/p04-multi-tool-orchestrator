"""Executor: run a validated plan as a DAG with parallel stages.

Conflict resolution: independent steps whose tools write the same
resource would race if run in parallel. The executor detects write-write
conflicts within a stage and serializes them (deterministic order by
plan index) while still parallelizing everything else. Read-write pairs
are also serialized. Threads are used for genuinely parallel stages; all
shared state is lock-protected.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .permissions import PolicyDB
from .planner import Plan, PlanStep
from .registry import ToolRegistry


@dataclass(frozen=True)
class ConflictResolution:
    """Record of one serialization decision made during scheduling."""

    step_a: str
    step_b: str
    resource: str
    reason: str  # "write-write" | "read-write"


@dataclass
class StepOutcome:
    name: str
    tool: str
    status: str  # "ok" | "error" | "denied"
    result: Any = None
    error: str | None = None
    seconds: float = 0.0
    stage: int = -1


@dataclass
class ExecutionReport:
    plan_steps: int
    outcomes: list[StepOutcome] = field(default_factory=list)
    conflicts_resolved: list[ConflictResolution] = field(default_factory=list)
    stages: int = 0

    @property
    def ok(self) -> bool:
        return all(o.status == "ok" for o in self.outcomes) and len(self.outcomes) == self.plan_steps

    def by_name(self, name: str) -> StepOutcome:
        for o in self.outcomes:
            if o.name == name:
                return o
        raise KeyError(name)


@dataclass
class Orchestrator:
    registry: ToolRegistry
    policies: PolicyDB
    max_workers: int = 4
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---------------------------------------------------------- scheduling
    def _stages(self, plan: Plan) -> list[list[PlanStep]]:
        """Topological stages: stage k runs after everything stage <k.

        Within a stage, write-conflicting steps are serialized out into
        their own follow-up mini-stages (conflict resolution).
        """
        remaining = {s.name: s for s in plan.steps}
        deps = {s.name: set(s.depends_on) for s in plan.steps}
        stages: list[list[PlanStep]] = []
        placed: set[str] = set()

        while remaining:
            ready = [n for n in remaining if not (deps[n] - placed)]
            if not ready:
                raise ValueError(f"unschedulable/cyclic plan: {sorted(remaining)}")
            ready.sort(key=lambda n: [s.name for s in plan.steps].index(n))

            # conflict-aware split of the ready set
            batch: list[PlanStep] = []
            busy_writers: set[str] = set()
            busy_all: set[str] = set()  # resources touched by batch members
            deferred: list[PlanStep] = []

            for name in ready:
                step = remaining[name]
                spec = self.registry.get(step.tool)
                writes = set(r for cap, r in zip(spec.capabilities, spec.resources)
                             if cap.startswith("write")) if spec.resources else set()
                touches = set(spec.resources) if spec.resources else set()

                conflict = None
                for r in touches:
                    if r in busy_writers:  # read-after-write or write-after-write
                        conflict = r
                        break
                    if r in busy_all and writes:  # write-after-read
                        conflict = r
                        break
                if conflict is None and writes & busy_all:
                    conflict = sorted(writes & busy_all)[0]

                if conflict is not None or (not spec.parallelizable and batch):
                    if conflict is not None:
                        holder = next(b for b in batch
                                      if conflict in set(self.registry.get(b.tool).resources or ()))
                        self._record_conflict(holder, step, conflict)
                    deferred.append(step)
                else:
                    batch.append(step)
                    busy_writers |= writes
                    busy_all |= touches

            for s in batch:
                placed.add(s.name)
                del remaining[s.name]
            stages.append(batch)
            if not batch and deferred:
                # non-parallelizable or conflicted-only stage: run one by one
                nxt = deferred[0]
                if not self.registry.get(nxt.tool).parallelizable:
                    placed.add(nxt.name)
                    del remaining[nxt.name]
                    stages.append([nxt])
                    deferred = deferred[1:]

        return stages

    def _record_conflict(self, a: PlanStep, b: PlanStep, resource: str) -> None:
        a_caps = self.registry.get(a.tool).capabilities
        b_caps = self.registry.get(b.tool).capabilities
        both_write = any(c.startswith("write") for c in a_caps) and \
                     any(c.startswith("write") for c in b_caps)
        reason = "write-write" if both_write else "read-write"
        self._conflicts.append(ConflictResolution(a.name, b.name, resource, reason))

    _conflicts: list[ConflictResolution] = field(default_factory=list)

    # ------------------------------------------------------------ execute
    def execute(self, plan: Plan, role: str = "operator") -> ExecutionReport:
        """Run the plan under `role`'s permissions. Fail-fast=False:
        denied/failed steps are recorded; dependents of failed steps are
        skipped with status 'error'."""
        report = ExecutionReport(plan_steps=len(plan.steps))
        results: dict[str, StepOutcome] = {}
        done_ok: set[str] = set()

        try:
            stages = self._stages(plan)
        except ValueError as exc:
            for s in plan.steps:
                report.outcomes.append(
                    StepOutcome(s.name, s.tool, "error", error=str(exc)))
            return report
        report.stages = len(stages)

        for stage_idx, stage in enumerate(stages):
            def run_step(step: PlanStep) -> StepOutcome:
                spec = self.registry.get(step.tool)
                t0 = time.time()
                try:
                    self.policies.check(role, spec)
                except Exception as exc:
                    return StepOutcome(step.name, step.tool, "denied",
                                       error=str(exc), seconds=time.time() - t0,
                                       stage=stage_idx)
                if set(step.depends_on) - done_ok:
                    missing = sorted(set(step.depends_on) - done_ok)
                    return StepOutcome(step.name, step.tool, "error",
                                       error=f"skipped: dependencies failed: {missing}",
                                       seconds=time.time() - t0, stage=stage_idx)
                try:
                    result = spec.fn(step.args)
                    return StepOutcome(step.name, step.tool, "ok", result=result,
                                       seconds=time.time() - t0, stage=stage_idx)
                except Exception as exc:
                    return StepOutcome(step.name, step.tool, "error", error=str(exc),
                                       seconds=time.time() - t0, stage=stage_idx)

            if len(stage) == 1:
                outcomes = [run_step(stage[0])]
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    outcomes = list(pool.map(run_step, stage))

            for o in outcomes:
                report.outcomes.append(o)
                results[o.name] = o
                if o.status == "ok":
                    done_ok.add(o.name)

        report.conflicts_resolved = list(self._conflicts)
        return report
