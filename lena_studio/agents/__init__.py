"""Sub-agent package for the Lena manager.

Importing this package registers every sub-agent module below into AGENT_REGISTRY
(base.py). To add a new sub-agent later:

  1. Write `lena_studio/agents/my_new_agent.py` with a `run(batch, task) -> dict`
     function. `task["input"]` holds whatever the planner (or the manager, for the
     review/publish stages) put there; return a plain dict — it's stored as
     `task["output"]`.
  2. At the bottom of that module, call:
       register_agent("my_new_agent", "generate", "one-line description", run,
                       needs_review=True, max_attempts=2)
     `stage` is one of plan | generate | review | publish. `needs_review=True` means a
     successful run parks the task at NEEDS_REVIEW for a human/Claude approval before
     it counts toward advancing the batch.
  3. Add one import line below: `from . import my_new_agent  # noqa: F401`.

The manager (manager.py) never needs to change — it dispatches purely by agent name and
by `by_stage(...)`, so registering the new module is the whole integration.
"""

from .base import AGENT_REGISTRY, AgentError, register_agent, get, by_stage, list_agents

from . import planner_agent   # noqa: F401 - registers "plan_batch"
from . import copy_agent      # noqa: F401 - registers "draft_copy"
from . import qa_agent        # noqa: F401 - registers "qa_check"
from . import publish_agent   # noqa: F401 - registers "publish_stub"

__all__ = ["AGENT_REGISTRY", "AgentError", "register_agent", "get", "by_stage", "list_agents"]
