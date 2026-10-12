"""Multi-tool orchestrator agent.

Dynamic tool registry, capability-based routing, permission scoping,
parallel execution with conflict resolution.

Modules:
- registry.py    dynamic registration, metadata, capability declarations
- permissions.py role-based permission scoping (allowlists per role)
- planner.py     LLM-driven plan: select tools + args, mark parallelizable
- executor.py    DAG execution: parallel stages, write-conflict detection
                 and resolution (serialize same-resource writers)
"""

from .executor import ConflictResolution, ExecutionReport, Orchestrator
from .permissions import PermissionError, Role, PolicyDB
from .planner import Plan, PlanStep, Planner
from .registry import ToolSpec, ToolRegistry

__all__ = [
    "Orchestrator", "ExecutionReport", "ConflictResolution",
    "Role", "PermissionError", "PolicyDB",
    "Plan", "PlanStep", "Planner", "ToolSpec", "ToolRegistry",
]
__version__ = "1.0.0"
