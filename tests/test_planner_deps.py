"""Extra planner tests: positional-index dependency handling."""

import pytest

from orchestrator import Planner, ToolRegistry, ToolSpec
from orchestrator.planner import PlanError


def _registry():
    r = ToolRegistry()
    r.register(ToolSpec("fetch_orders", "reads orders", lambda a: a,
                        ("read:orders",), ("db.orders",)))
    r.register(ToolSpec("create_report", "writes report", lambda a: a,
                        ("write:reports",), ("db.reports",)))
    return r


def _planner(raw):
    return Planner(_registry(), llm=lambda m: raw)


class TestIndexDeps:
    def test_index_dependencies_resolved(self):
        p = _planner(
            '[{"tool": "fetch_orders", "args": {}},'
            ' {"tool": "create_report", "args": {}, "depends_on": [0]}]'
        ).plan("x")
        assert p.steps[1].depends_on == ["step-0"]

    def test_self_dependency_rejected(self):
        with pytest.raises(PlanError):
            _planner('[{"name": "a", "tool": "fetch_orders", "depends_on": [0]}]').plan("x")

    def test_out_of_range_index_rejected(self):
        with pytest.raises(PlanError):
            _planner(
                '[{"tool": "fetch_orders"},'
                ' {"tool": "create_report", "depends_on": [5]}]'
            ).plan("x")

    def test_named_deps_still_work(self):
        p = _planner(
            '[{"name": "f", "tool": "fetch_orders"},'
            ' {"name": "r", "tool": "create_report", "depends_on": ["f"]}]'
        ).plan("x")
        assert p.steps[1].depends_on == ["f"]
