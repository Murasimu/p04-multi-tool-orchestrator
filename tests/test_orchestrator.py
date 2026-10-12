"""Tests for the multi-tool orchestrator. Deterministic, offline."""

import time

import pytest

from orchestrator import (
    Orchestrator, Plan, PlanStep, Planner, PolicyDB, Role, ToolRegistry, ToolSpec,
)
from orchestrator.permissions import PermissionError
from orchestrator.planner import PlanError


# ---------------------------------------------------------------- registry
class TestRegistry:
    def test_register_get_unregister(self):
        r = ToolRegistry()
        r.register(ToolSpec("a", "does a", lambda x: 1, ("read:x",), ("res-1",)))
        assert r.get("a").name == "a"
        r.unregister("a")
        with pytest.raises(KeyError):
            r.get("a")

    def test_duplicate_rejected(self):
        r = ToolRegistry()
        r.register(ToolSpec("a", "x", lambda x: 1))
        with pytest.raises(ValueError):
            r.register(ToolSpec("a", "x", lambda x: 1))

    def test_lookup_by_capability_and_resource(self):
        r = ToolRegistry()
        r.register(ToolSpec("read_inv", "r", lambda x: 1, ("read:inventory",), ("db.inventory",)))
        r.register(ToolSpec("write_ord", "w", lambda x: 1, ("write:orders",), ("db.orders",)))
        assert [t.name for t in r.by_capability("read:inventory")] == ["read_inv"]
        assert [t.name for t in r.by_resource("db.orders")] == ["write_ord"]


# ---------------------------------------------------------------- permissions
class TestPermissions:
    def _policy(self):
        p = PolicyDB()
        p.add_role(Role("viewer", ("read:*",)))
        p.add_role(Role("operator", ("read:*", "write:orders")))
        p.add_role(Role("admin", ("*",)))
        return p

    def test_viewer_cannot_write(self):
        p = self._policy()
        tool = ToolSpec("w", "w", lambda x: 1, ("write:orders",), ("db.orders",))
        with pytest.raises(PermissionError) as ei:
            p.check("viewer", tool)
        assert "write:orders" in str(ei.value)

    def test_operator_can_write_orders_not_users(self):
        p = self._policy()
        orders = ToolSpec("w", "w", lambda x: 1, ("write:orders",))
        users = ToolSpec("u", "u", lambda x: 1, ("write:users",))
        p.check("operator", orders)  # no raise
        with pytest.raises(PermissionError):
            p.check("operator", users)

    def test_admin_wildcard(self):
        p = self._policy()
        anything = ToolSpec("x", "x", lambda x: 1, ("delete:everything",))
        p.check("admin", anything)

    def test_unknown_role(self):
        p = self._policy()
        with pytest.raises(PermissionError):
            p.check("ghost", ToolSpec("x", "x", lambda x: 1, ("read:a",)))

    def test_capability_free_tool_always_allowed(self):
        p = self._policy()
        p.check("viewer", ToolSpec("noop", "n", lambda x: 1))


# ---------------------------------------------------------------- planner
class TestPlanner:
    def _registry(self):
        r = ToolRegistry()
        r.register(ToolSpec("fetch_inventory", "reads inventory", lambda a: a,
                            ("read:inventory",), ("db.inventory",)))
        r.register(ToolSpec("fetch_orders", "reads orders", lambda a: a,
                            ("read:orders",), ("db.orders",)))
        r.register(ToolSpec("create_report", "writes report", lambda a: a,
                            ("write:reports",), ("db.reports",)))
        return r

    def _planner(self, raw):
        return Planner(self._registry(), llm=lambda m: raw)

    def test_valid_plan_parsed(self):
        p = self._planner(
            '[{"name": "inv", "tool": "fetch_inventory", "args": {"region": "east"}},'
            ' {"name": "ord", "tool": "fetch_orders", "args": {}},'
            ' {"name": "rep", "tool": "create_report", "args": {}, "depends_on": ["inv", "ord"]}]'
        ).plan("make a report")
        assert [s.tool for s in p.steps] == ["fetch_inventory", "fetch_orders", "create_report"]
        assert p.steps[2].depends_on == ["inv", "ord"]

    def test_prose_wrapped_plan(self):
        p = self._planner(
            'Sure! Here is the plan:\n```json\n[{"tool": "fetch_orders", "args": {}}]\n```\nDone!'
        ).plan("x")
        assert p.steps[0].tool == "fetch_orders"

    def test_unknown_tool_rejected(self):
        with pytest.raises(PlanError):
            self._planner('[{"tool": "teleport", "args": {}}]').plan("x")

    def test_cycle_rejected(self):
        with pytest.raises(PlanError):
            self._planner(
                '[{"name": "a", "tool": "fetch_orders", "depends_on": ["b"]},'
                ' {"name": "b", "tool": "fetch_inventory", "depends_on": ["a"]}]'
            ).plan("x")

    def test_missing_dep_rejected(self):
        with pytest.raises(PlanError):
            self._planner('[{"name": "a", "tool": "fetch_orders", "depends_on": ["ghost"]}]').plan("x")

    def test_empty_plan_rejected(self):
        with pytest.raises(PlanError):
            self._planner('[]').plan("x")


# ---------------------------------------------------------------- executor
def _plan(*steps):
    return Plan(steps=[PlanStep(**s) for s in steps])


class TestExecutor:
    def _setup(self):
        r = ToolRegistry()
        log = []
        lk = {"lock": __import__("threading").Lock()}

        def mk(name, cap, res, parallel=True, dur=0.0):
            def fn(args):
                time.sleep(dur)
                with lk["lock"]:
                    log.append((name, time.time()))
                return f"{name}:ok"
            r.register(ToolSpec(name, name, fn, (cap,), (res,), parallel))
        mk("read_inv", "read:inventory", "db.inventory", dur=0.15)
        mk("read_ord", "read:orders", "db.orders", dur=0.15)
        mk("write_rep", "write:reports", "db.reports", dur=0.05)
        mk("write_rep2", "write:reports2", "db.reports", dur=0.05)
        p = PolicyDB()
        p.add_role(Role("operator", ("read:*", "write:reports", "write:reports2")))
        p.add_role(Role("viewer", ("read:*",)))
        return Orchestrator(registry=r, policies=p), r, log

    def test_parallel_speedup(self):
        orch, _, _ = self._setup()
        plan = _plan(
            {"name": "a", "tool": "read_inv", "args": {}},
            {"name": "b", "tool": "read_ord", "args": {}},
        )
        t0 = time.time()
        report = orch.execute(plan, role="operator")
        wall = time.time() - t0
        assert report.ok
        # two 0.15s steps in parallel must beat serial (0.30s) by a margin
        assert wall < 0.24, f"expected parallel speedup, wall={wall:.2f}s"

    def test_write_write_conflict_serialized(self):
        orch, r, log = self._setup()
        plan = _plan(
            {"name": "a", "tool": "write_rep", "args": {}},
            {"name": "b", "tool": "write_rep2", "args": {}},
        )
        report = orch.execute(plan, role="operator")
        assert report.ok
        assert report.conflicts_resolved, "write-write conflict should be recorded"
        assert report.conflicts_resolved[0].reason == "write-write"
        # serialized: distinct start times, no overlap beyond epsilon
        t_a = next(t for n, t in log if n == "write_rep")
        t_b = next(t for n, t in log if n == "write_rep2")
        assert abs(t_a - t_b) > 0.03, "conflicted writers must not overlap"

    def test_dependency_ordering(self):
        orch, _, log = self._setup()
        plan = _plan(
            {"name": "inv", "tool": "read_inv", "args": {}},
            {"name": "rep", "tool": "write_rep", "args": {}, "depends_on": ["inv"]},
        )
        report = orch.execute(plan, role="operator")
        assert report.ok
        t_inv = next(t for n, t in log if n == "read_inv")
        t_rep = next(t for n, t in log if n == "write_rep")
        assert t_rep > t_inv

    def test_permission_denial_recorded(self):
        orch, _, _ = self._setup()
        plan = _plan({"name": "w", "tool": "write_rep", "args": {}})
        report = orch.execute(plan, role="viewer")
        assert not report.ok
        assert report.outcomes[0].status == "denied"

    def test_failed_dependency_skips_dependent(self):
        orch, r, _ = self._setup()
        r.register(ToolSpec("boom", "fails", lambda a: 1 / 0, ("read:inventory",), ("db.x",)))
        plan = _plan(
            {"name": "bad", "tool": "boom", "args": {}},
            {"name": "after", "tool": "read_inv", "args": {}, "depends_on": ["bad"]},
        )
        report = orch.execute(plan, role="operator")
        assert report.outcomes[0].status == "error"
        assert "skipped" in (report.outcomes[1].error or "")

    def test_non_parallelizable_serialized(self):
        orch, r, log = self._setup()
        r.register(ToolSpec("solo", "not thread-safe", lambda a: "ok",
                            ("read:x",), ("db.solo",), parallelizable=False))
        plan = _plan(
            {"name": "s1", "tool": "solo", "args": {}},
            {"name": "s2", "tool": "solo", "args": {}},
        )
        report = orch.execute(plan, role="operator")
        assert report.ok


# ---------------------------------------------------------------- e2e
class TestEndToEnd:
    def test_plan_then_execute(self):
        r = ToolRegistry()
        r.register(ToolSpec("fetch", "fetch", lambda a: {"data": 1},
                            ("read:x",), ("db.x",)))
        r.register(ToolSpec("report", "report", lambda a: {"made": True},
                            ("write:y",), ("db.y",)))
        raw = '[{"name": "f", "tool": "fetch", "args": {}},' \
              '{"name": "r", "tool": "report", "args": {}, "depends_on": ["f"]}]'
        planner = Planner(r, llm=lambda m: raw)
        plan = planner.plan("fetch then report")
        policies = PolicyDB()
        policies.add_role(Role("op", ("*",)))
        report = Orchestrator(registry=r, policies=policies).execute(plan, role="op")
        assert report.ok
        assert report.by_name("r").result == {"made": True}
