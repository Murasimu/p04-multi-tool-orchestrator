# Multi-Tool Orchestrator Agent

**Project 4 / 12 — Agentic AI portfolio.**
*Shows: you can coordinate complex workflows.*

An orchestrator that turns a natural-language task into a validated tool
DAG and executes it safely: dynamic tool registry, capability-based
routing, role-based permission scoping, parallel execution with automatic
write-conflict resolution.

## Why this matters

Agents fail in coordination, not in single tool calls: steps race on
shared resources, the wrong role runs privileged tools, LLM plans
reference tools that don't exist or form dependency cycles, and
everything runs serially "to be safe". This project addresses each:

1. **Dynamic registry** — tools register at runtime with capabilities
   (`read:inventory`, `write:orders`) and resources (`db.inventory`).
2. **Capability routing** — the planner sees the catalog and can query by
   capability or resource.
3. **Permission scoping** — roles hold capability patterns (`read:*`,
   `write:orders`); every step is checked before execution and denials
   are explicit, audit-friendly exceptions.
4. **Parallel execution + conflict resolution** — independent steps run
   in a thread pool; within a stage, steps touching the same resource are
   serialized (write-write and read-write pairs) and every serialization
   is logged with a reason.

## Architecture

```
task ──> Planner(LLM) ──> raw JSON plan
                │
        structural validation: tools exist, deps resolvable
        (names / indices / tool names), DAG acyclic, size capped
                │
             Plan (list of steps with depends_on)
                │
        Orchestrator.execute(plan, role)
                │
        stage scheduling (topological + conflict-aware)
                │
   ┌────────────┴────────────┐
   │ stage 0: [a, b, c]      │  parallel (ThreadPool)
   │ stage 1: [d]            │  after its deps
   │ ...                     │  write-conflicts serialized + logged
   └─────────────────────────┘
                │
        ExecutionReport: per-step status/result/timing,
        conflicts_resolved, stages
```

- `orchestrator/registry.py` — ToolSpec + dynamic registry
- `orchestrator/permissions.py` — Role / PolicyDB with pattern allowlists
- `orchestrator/planner.py` — LLM plan proposal + structural validation
  (unknown tools, cycles, self-deps, out-of-range indices all rejected)
- `orchestrator/executor.py` — stage scheduler, parallelism, conflict
  resolution, per-step outcomes with denial/skip/error states

## Quickstart

```bash
uv pip install -e ".[demo]"
python demo.py            # scripted planner, no network
python demo.py --live     # live LLM generates the plan
```

Demo scenario: a fictional e-commerce back office. Three reads run in
parallel (0.15s each, ~0.2s wall vs 0.5s+ serial), the report is
generated once the reads finish, a viewer role gets the write steps
denied, and two audit writers on the same resource are serialized with a
logged conflict.

## Tests

```bash
pytest -q
```

25 tests cover: registry CRUD + lookups, permission patterns (wildcards,
role separation, unknown roles, capability-free tools), planner
validation (valid plans, prose-wrapped JSON, unknown tools, cycles,
missing deps, empty plans, positional-index deps, self-deps, out-of-range
indices), executor behavior (measured parallel speedup, write-write
conflict serialization with timing proof, dependency ordering, denial
recording, failed-dependency skips, non-parallelizable tools), and a
full plan→execute end-to-end run.

## Design decisions

- **Validate plans structurally, not hopefully** — LLM plans are data,
  so they get the same treatment as user input: reject unknown tools,
  cycles, and unresolvable dependencies before anything executes.
- **Deps may be names, indices, or tool names** — live models use all
  three; the planner normalizes instead of failing.
- **Conflicts resolved at scheduling, not with a big lock** — a global
  lock destroys parallelism (this exact bug was caught by the
  parallel-speedup test); instead same-resource steps within a stage are
  split into ordered sub-stages and the decision is recorded.
- **Fail-open reads, fail-closed writes** — permission patterns make the
  read/write split explicit, so a viewer can aggregate but never mutate.
- **Denials and skips are outcomes, not exceptions** — the report shows
  per-step `ok | error | denied`, so partial failures are debuggable.

## Portfolio context

Project 4 of a 12-project agentic-AI portfolio (structured output, RAG
grounding, ReAct planning, tool orchestration, memory, HITL approval,
cost routing, event automation, debate, self-eval, observability, OSS
contribution). See the profile README for the full map.
