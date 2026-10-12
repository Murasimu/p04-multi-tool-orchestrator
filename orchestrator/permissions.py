"""Permission scoping: which roles may run which tools.

A tool's `capabilities` are the verbs it performs; a Role holds an
allowlist of capability patterns like "read:*", "write:orders". Execution
is checked before anything runs, and denials are explicit exceptions with
the reason attached (audit-friendly), never silent skips.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .registry import ToolSpec


class PermissionError(Exception):
    """A role is not allowed to run a tool."""

    def __init__(self, role: str, tool: str, missing_capability: str):
        self.role = role
        self.tool = tool
        self.missing_capability = missing_capability
        super().__init__(
            f"role {role!r} may not run {tool!r}: "
            f"missing capability {missing_capability!r}"
        )


@dataclass(frozen=True)
class Role:
    name: str
    allowed: tuple[str, ...]  # capability patterns

    def permits(self, capability: str) -> bool:
        for pattern in self.allowed:
            if pattern == "*":
                return True
            if pattern.endswith(":*"):
                if capability.startswith(pattern[:-1]):
                    return True
            elif pattern == capability:
                return True
        return False


@dataclass
class PolicyDB:
    roles: dict[str, Role] = field(default_factory=dict)

    def add_role(self, role: Role) -> None:
        self.roles[role.name] = role

    def check(self, role_name: str, tool: ToolSpec) -> None:
        """Raise PermissionError if `role_name` cannot run `tool`."""
        role = self.roles.get(role_name)
        if role is None:
            raise PermissionError(role_name, tool.name, "<role-does-not-exist>")
        if not tool.capabilities:
            return  # capability-free tools are always permitted
        for cap in tool.capabilities:
            if role.permits(cap):
                return
        raise PermissionError(role_name, tool.name, tool.capabilities[0])
