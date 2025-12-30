"""
Extended state definition for the full MARS swarm.
Replaces the minimal BasicAgentState with complete structure.
"""

# imports
from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional
from langchain_core.messages import AnyMessage
import operator

# helper classes (none needed here)

class MARSState(TypedDict):
    """
    Complete state used by the LangGraph workflow of the MARS swarm.
    Contains conversation history, core reasoning frontier, all thoughts,
    pending injections, and routing information.
    """
    messages: Annotated[List[AnyMessage], operator.add]                  # running conversation log
    core_context: Dict[str, Any]                                         # current frontier Thought (dict form)
    thoughts: List[Dict[str, Any]]                                       # all active atomic Thoughts
    injection_queue: List[Dict[str, Any]]                                # Thoughts queued for merging
    active_agent: Literal["orchestrator", "summary", "thought_generator", "memory_swarm"]
    last_handoff_reason: Optional[str]                                   # reason for last handoff (debugging)