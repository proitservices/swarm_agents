"""
Shared types and utilities for MARS swarm system
Central place for all type definitions (TypedDict) and helper factories.
Ensures consistency across orchestrator, memory agents, summary, thought generator, etc.

File location: ./mars/types.py
"""

# imports
from typing import TypedDict, List, Dict, Any, Literal, Annotated, Optional
from langchain_core.messages import AnyMessage
import operator
from datetime import datetime
from uuid import uuid4

# ──────────────────────────────────────────────────────────────────────────────
# Helper classes
# (none needed - we use plain TypedDict for serialization simplicity)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Operational classes / factories
# ──────────────────────────────────────────────────────────────────────────────

def create_thought(
    narrative: str,
    meta_narrative: str = "",
    origins: List[str] = None,
    relations: List[Dict[str, str]] = None,
    initial_relevance: float = 0.5,
    is_seed: bool = False
) -> 'Thought':
    """
    Factory function to create a new consistent Thought structure.

    This is the single entry point for both seed thoughts and dynamically generated ones.
    Generated thoughts should have is_seed=False (default), seed thoughts should pass is_seed=True.

    Args:
        narrative (str): Main descriptive content of the thought
        meta_narrative (str): Summary of scope, dependencies and purpose
        origins (List[str], optional): List of predecessor thought_ids
        relations (List[Dict[str, str]], optional): Semantic relations
        initial_relevance (float, optional): Starting relevance score (0.0-1.0)
        is_seed (bool, optional): True if this is a static seed thought, False for generated

    Returns:
        Thought: Fully formed, timestamped thought dictionary
    """
    if origins is None:
        origins = []
    if relations is None:
        relations = []

    prefix = "prime" if is_seed else "gen"
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    random_part = str(uuid4())[:8]
    thought_id = f"{prefix}-{timestamp}-{random_part}"

    return {
        "thought_timestamp": datetime.utcnow().isoformat(),
        "thought_id": thought_id,
        "origin_thought_ids": origins,
        "relations": relations,
        "narrative": narrative.strip(),
        "meta_narrative": meta_narrative.strip(),
        "relevance_score": float(initial_relevance),
        "is_seed": is_seed,
        "last_evaluated": None  # Will be set during first evaluation
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core class that runs the main logic → here we define types only
# ──────────────────────────────────────────────────────────────────────────────

class Thought(TypedDict):
    """
    Atomic unit of memory/knowledge in the MARS swarm.
    All fields are JSON-serializable.
    """
    thought_timestamp: str
    """ISO timestamp when thought was created"""
    thought_id: str
    """Unique identifier (prime-... for seeds, gen-... for generated)"""
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
    is_seed: bool
    """Flag: True for static seed thoughts, False for dynamically generated"""
    last_evaluated: Optional[str]
    """ISO timestamp of last evaluation (set by swarm)"""


class MARSState(TypedDict):
    """
    Complete state schema for LangGraph graph execution.
    Uses Annotated + operator.add for proper messages accumulation.
    """
    messages: Annotated[List[AnyMessage], operator.add]
    """Accumulated conversation history (Human/AI/System/Tool messages)"""
    core_context: Dict[str, Any]
    """Current working context / frontier thought"""
    thoughts: List[Thought]
    """All active, tracked thoughts in the swarm"""
    injection_queue: List[Thought]
    """Thoughts queued for context injection (usually topo-sorted)"""
    active_agent: Literal["orchestrator", "summary", "thought_generator", "memory_swarm"]
    """Name of the last/currently active processing unit"""
    last_handoff_reason: Optional[str]
    """Human-readable reason for the most recent agent transition"""


# ──────────────────────────────────────────────────────────────────────────────
# Quick usage example (for documentation/testing)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example of a generated thought
    gen_thought = create_thought(
        narrative="User asked about tort law principles in New York state.",
        meta_narrative="Initial query parsing - domain: legal, jurisdiction: NY, topic: torts",
        origins=[],
        initial_relevance=0.65,
        is_seed=False
    )
    print("Generated thought ID:", gen_thought["thought_id"])

    # Example of a seed thought
    seed_thought = create_thought(
        narrative="Always remain precise and jurisdiction-aware in legal reasoning.",
        meta_narrative="Core legal persona - guiding principle",
        origins=[],
        initial_relevance=0.92,
        is_seed=True
    )
    print("Seed thought ID:", seed_thought["thought_id"])