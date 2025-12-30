"""
MARS Swarm Core Graph Builder & Runner

This module defines the central LangGraph structure for the MARS swarm.
It coordinates all agent types (orchestrator, summary, thought_generator, memory_swarm)
and handles conditional routing based on relevance and injection queue.

File location: mars/core/graph.py
"""

# imports
from typing import Dict, Any, Callable, List, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from mars.types import MARSState, Thought
from mars.agents.orchestrator.agent import create_orchestrator_node
from mars.agents.summary.agent import create_summary_node
from mars.agents.thought_generator.agent import create_thought_generator_node
from mars.agents.memory.swarm import create_memory_swarm_node


# ──────────────────────────────────────────────────────────────────────────────
# Helper classes & utilities
# ──────────────────────────────────────────────────────────────────────────────

class StaticRelevanceEvaluator:
    """
    Temporary evaluator until proper CoT-based relevance scoring is implemented.
    Uses fixed threshold as requested.
    """
    THRESHOLD = 0.85

    @staticmethod
    def is_relevant(thought: Thought) -> bool:
        """Simple static threshold check."""
        return thought.get("relevance_score", 0.0) >= StaticRelevanceEvaluator.THRESHOLD


def route_after_memory_swarm(state: MARSState) -> str:
    """
    Conditional router after memory swarm execution.
    Decides whether to go to thought_generator or summary.

    Args:
        state: Current MARS state

    Returns:
        str: Next node name ("thought_generator" or "summary")
    """
    if not state.get("thoughts"):
        return "summary"  # initial flow - no thoughts yet

    last_thought = state["thoughts"][-1]

    # If last thought is low relevance → generate new thought
    if last_thought["relevance_score"] < StaticRelevanceEvaluator.THRESHOLD:
        return "thought_generator"

    # If we have queued injections → merge them
    if state.get("injection_queue"):
        return "summary"

    # Default: back to summary / orchestrator
    return "summary"


# ──────────────────────────────────────────────────────────────────────────────
# Core class
# ──────────────────────────────────────────────────────────────────────────────

class MARSGraph:
    """
    Builds and compiles the LangGraph workflow for the entire MARS swarm.
    Provides compiled runnable application.
    """

    def __init__(self, llm):
        self.llm = llm
        self.checkpointer = MemorySaver()  # simple in-memory persistence
        self.app = self._build()

    def _build(self):
        """Internal method - constructs the graph structure."""
        builder = StateGraph(MARSState)

        # Nodes - each agent type gets its own node
        builder.add_node("orchestrator",      create_orchestrator_node(self.llm))
        builder.add_node("summary",           create_summary_node(self.llm))
        builder.add_node("thought_generator", create_thought_generator_node(self.llm))
        builder.add_node("memory_swarm",      create_memory_swarm_node(self.llm))

        # Entry point: always start with orchestrator
        builder.set_entry_point("orchestrator")

        # Main flow edges
        builder.add_edge("orchestrator", "memory_swarm")      # broadcast after reasoning step
        builder.add_edge("thought_generator", "memory_swarm") # new thought → back to swarm

        # After memory swarm → conditional routing
        builder.add_conditional_edges(
            source="memory_swarm",
            path=route_after_memory_swarm,
            path_map={
                "thought_generator": "thought_generator",
                "summary": "summary"
            }
        )

        # After summary → back to orchestrator (merged context)
        builder.add_edge("summary", "orchestrator")

        # For now: cycle ends after orchestrator (later: add proper termination condition)
        builder.add_edge("orchestrator", END)

        # Compile with checkpointing
        return builder.compile(checkpointer=self.checkpointer)

    def invoke(self, initial_state: MARSState, config: Optional[Dict] = None) -> MARSState:
        """
        Execute the full graph workflow synchronously.
        Now properly forwards config to the underlying app.
        """
        return self.app.invoke(initial_state, config=config)

    def stream(self, initial_state: MARSState):
        """
        Stream graph execution events (useful for real-time logging / UI).

        Args:
            initial_state: Starting state

        Yields:
            Dict: Graph execution events/chunks
        """
        for event in self.app.stream(initial_state):
            yield event