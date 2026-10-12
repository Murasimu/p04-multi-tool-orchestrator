"""Dynamic tool registry.

Tools declare:
- name, description
- capabilities: what they can do (e.g. "read:inventory", "write:orders")
- resources: what they touch (e.g. "db.inventory", "api.crm")
- parallelizable: whether it is safe to run concurrently

The registry is dynamic: register/unregister at runtime, list by
capability or resource. No import-time coupling -- anything callable can
be a tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    fn: Callable[[dict[str, Any]], Any]
    capabilities: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    parallelizable: bool = True


class UnknownToolError(KeyError):
    pass


@dataclass
class ToolRegistry:
    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self.tools:
            raise ValueError(f"tool {spec.name!r} already registered")
        self.tools[spec.name] = spec

    def unregister(self, name: str) -> None:
        if name not in self.tools:
            raise UnknownToolError(name)
        del self.tools[name]

    def get(self, name: str) -> ToolSpec:
        if name not in self.tools:
            raise UnknownToolError(name)
        return self.tools[name]

    def by_capability(self, capability: str) -> list[ToolSpec]:
        return [t for t in self.tools.values() if capability in t.capabilities]

    def by_resource(self, resource: str) -> list[ToolSpec]:
        return [t for t in self.tools.values() if resource in t.resources]

    def describe(self) -> str:
        return "\n".join(
            f"- {t.name}: {t.description} "
            f"[caps={','.join(t.capabilities) or '-'}; "
            f"resources={','.join(t.resources) or '-'}; "
            f"parallel={'yes' if t.parallelizable else 'no'}]"
            for t in self.tools.values()
        )
