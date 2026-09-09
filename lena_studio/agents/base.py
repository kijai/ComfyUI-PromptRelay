"""Sub-agent registry — the workers the Lena manager delegates to.

Every sub-agent is a plain function `run(batch, task) -> dict`, registered once via
`register_agent()`. The manager looks agents up purely by name (`task["agent"]`) and by
pipeline stage (`by_stage("generate")`, etc.) — it never imports a specific agent
module, so adding a new capability later means writing one function and one
registration call. See the package docstring in `__init__.py` for the exact steps.
"""

AGENT_REGISTRY = {}


class AgentError(Exception):
    """Raised by a sub-agent (or the manager) for an expected failure — bad input, a
    backend that's down, a missing precondition. The manager catches this (and any
    other exception a sub-agent raises) and marks the task FAILED/retried instead of
    crashing the batch."""


def register_agent(name, stage, description, fn, needs_review=True, max_attempts=2):
    """Register a sub-agent. `stage` is one of plan | generate | review | publish —
    the manager dispatches to agents by stage at each handoff point. `needs_review=True`
    means a successful run parks the task at NEEDS_REVIEW instead of DONE, so a
    human/Claude approval is required before it counts toward advancing the batch."""
    AGENT_REGISTRY[name] = {
        "name": name,
        "stage": stage,
        "description": description,
        "fn": fn,
        "needs_review": needs_review,
        "max_attempts": max_attempts,
    }
    return AGENT_REGISTRY[name]


def get(name):
    return AGENT_REGISTRY.get(name)


def by_stage(stage):
    return [a for a in AGENT_REGISTRY.values() if a["stage"] == stage]


def list_agents():
    """Registry summary for monitoring (GET /promptrelay/lena/agents)."""
    return [
        {"name": a["name"], "stage": a["stage"], "description": a["description"],
         "needs_review": a["needs_review"], "max_attempts": a["max_attempts"]}
        for a in AGENT_REGISTRY.values()
    ]
