"""
Shared types and utilities for MARS swarm system
File location: ./mars/types.py

Central place for all type definitions (TypedDict) and helper factories.
This ensures consistency across orchestrator, memory agents, summary, etc.
"""
# imports
from typing import TypedDict, List, Dict, Any, Literal, Annotated, Optional
from langchain_core.messages import AnyMessage
import operator
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Helper classes
# (none needed - we use plain TypedDict for serialization simplicity)
# ──────────────────────────────────────────────────────────────────────────────

# Operational classes / factories
def create_thought(
    narrative: str,
    meta_narrative: str,
    origins: List[str] = None,
    relations: List[Dict[str, str]] = None,
    initial_relevance: float = 0.0
) -> 'Thought':
    """
    Factory function to create a new consistent Thought structure.

    Args:
        narrative (str): Main descriptive content of the thought
        meta_narrative (str): Summary of scope, dependencies and purpose
        origins (List[str], optional): List of predecessor thought_ids
        relations (List[Dict[str, str]], optional): Semantic relations to other thoughts
        initial_relevance (float, optional): Starting relevance score (default 0.0)

    Returns:
        Thought: Fully formed, timestamped thought dictionary
    """
    if origins is None:
        origins = []
    if relations is None:
        relations = []

    now = int(datetime.now().timestamp())
    return {
        "thought_timestamp": now,
        "thought_id": f"thought-{now}",
        "origin_thought_ids": origins,
        "relations": relations,
        "narrative": narrative,
        "meta_narrative": meta_narrative,
        "relevance_score": initial_relevance
    }

# Core class that runs the main logic → here we define types only
class Thought(TypedDict):
    """
    Atomic unit of memory/knowledge in the MARS swarm.
    All fields are JSON-serializable.
    """
    thought_timestamp: int
    """Unix timestamp (seconds) when thought was created"""

    thought_id: str
    """Unique identifier, format: thought-<unix_timestamp>"""

    origin_thought_ids: List[str]
    """IDs of thoughts this one depends on / derives from"""

    relations: List[Dict[str, str]]
    """Semantic relations: [{"type": "supports|contradicts|refines", "target_id": "...", "reason": "..."}]"""

    narrative: str
    """Core descriptive content (usually 1-4 sentences)"""

    meta_narrative: str
    """High-level summary: scope, purpose, known dependencies"""

    relevance_score: float
    """Current relevance/confidence score (0.0-1.0), updated by memory agents"""


class MARSState(TypedDict):
    """
    Complete state schema for LangGraph graph execution.
    Uses Annotated + operator.add for proper messages accumulation.
    """
    messages: Annotated[List[AnyMessage], operator.add]
    """Accumulated conversation history (Human/AI/System/Tool messages)"""

    core_context: Dict[str, Any]
    """Current working context / frontier thought (often mirrors latest high-relevance thought)"""

    thoughts: List[Thought]
    """All active, tracked thoughts in the swarm"""

    injection_queue: List[Thought]
    """Thoughts queued for context injection (usually topo-sorted)"""

    active_agent: Literal["orchestrator", "summary", "thought_generator", "memory_swarm"]
    """Name of the last/currently active processing unit"""

    last_handoff_reason: Optional[str]
    """Human-readable reason for the most recent agent transition"""


# Quick usage example (for documentation/testing)
if __name__ == "__main__":
    initial_thought = create_thought(
        narrative="User asked about tort law principles in New York state.",
        meta_narrative="Initial query parsing - domain: legal, jurisdiction: NY, topic: torts",
        origins=[],
        relations=[{"type": "initiates", "target_id": "future-query-analysis", "reason": "starting point"}]
    )
    print("Example Thought created:", initial_thought["thought_id"])