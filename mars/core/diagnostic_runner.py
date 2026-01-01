"""
Diagnostic Runner for MARS Swarm - Manual step-by-step execution
Supports both mock and real LLM, preserves full state across nodes.
Now includes better visibility into LLM outputs and state changes.
"""

# imports
import time
from typing import Dict, Any, List

from langchain_core.messages import AIMessage, HumanMessage, AnyMessage

from mars.types import MARSState, create_thought
from mars.infrastructure.llm import create_llm
from mars.agents.orchestrator.agent import create_orchestrator_node
from mars.agents.summary.agent import create_summary_node
from mars.agents.memory.swarm import create_memory_swarm_node
from mars.config import AppConfig


# helper classes
class MockLLM:
    """Minimal mock fallback - not used when use_real_llm=True"""
    def invoke(self, input_data, **kwargs):
        return AIMessage(content=f"Mock response {time.strftime('%H:%M:%S')}")


# operational classes (none needed in diagnostic runner)


# helper functions
def merge_state_updates(original: MARSState, update: Dict[str, Any]) -> MARSState:
    """
    Safely merges node updates into full state without losing keys.

    Args:
        original (MARSState): Current complete state
        update (Dict[str, Any]): Delta returned by node

    Returns:
        MARSState: New merged state
    """
    new_state = original.copy()

    for key, value in update.items():
        if isinstance(value, list) and key in ("messages", "thoughts", "injection_queue"):
            new_state[key] = new_state.get(key, []) + value
        else:
            new_state[key] = value

    return new_state


def print_last_message(state: MARSState, label: str = "Last message"):
    """
    Prints the content of the most recent message with a label.

    Args:
        state (MARSState): Current state
        label (str): Descriptive label for context
    """
    messages = state.get("messages", [])
    if not messages:
        print(f"{label}: (no messages)")
        return

    last_msg = messages[-1]
    content = last_msg.content.strip()
    if len(content) > 280:
        content = content[:280] + "..."
    print(f"{label}:\n{content}\n")


# Core execution logic
def manual_graph_trace(
    config: AppConfig,
    use_real_llm: bool = True,
    query: str = "Summarize key principles of tort law in New York. Then propose a new law after careful consideration and deep thinking. When the new law is formed assess its application to a example case with your reasoning presented in detail."
) -> None:
    """
    Runs manual step-by-step execution of the swarm graph nodes.
    Prints LLM outputs and key state changes for visibility.

    Args:
        config (AppConfig): Application configuration
        use_real_llm (bool): Use real vLLM endpoint (default: True)
        query (str): Test query to process
    """
    print("=== MANUAL GRAPH TRACE ===")
    print(f"LLM: {'REAL vLLM' if use_real_llm else 'MOCK'}")
    print("Query:", query)
    print("Start:", time.strftime("%H:%M:%S"))

    llm = create_llm(config) if use_real_llm else MockLLM()

    # Seed thought
    seed = create_thought(
        narrative=f"Initial parsing of query: {query}",
        meta_narrative="Diagnostic runner seed · Legal domain · NY tort law",
        initial_relevance=0.65
    )

    state: MARSState = {
        "messages": [HumanMessage(content=query)],
        "core_context": {"topic": "Principles of tort law in New York"},
        "thoughts": [seed],
        "injection_queue": [],
        "active_agent": "orchestrator",
        "last_handoff_reason": None
    }

    print(f"Initial thoughts: {len(state['thoughts'])} (id: {state['thoughts'][0]['thought_id']})")

    # ─────────────── Orchestrator ───────────────
    print("\n=== ORCHESTRATOR ===")
    orch_node = create_orchestrator_node(llm)
    orch_update = orch_node(state)
    state = merge_state_updates(state, orch_update)
    print(f"Messages now: {len(state['messages'])}")
    print_last_message(state, "Orchestrator response")

    # ─────────────── Memory Swarm ───────────────
    print("\n=== MEMORY SWARM ===")
    swarm_node = create_memory_swarm_node(llm)
    swarm_update = swarm_node(state)
    state = merge_state_updates(state, swarm_update)
    print(f"Thoughts now: {len(state.get('thoughts', []))}")
    print(f"Injection queue size: {len(state.get('injection_queue', []))}")
    if state.get("injection_queue"):
        print("Queued thought IDs:", [t["thought_id"] for t in state["injection_queue"]])

    # ─────────────── Summary (if queue has content) ───────────────
    if state.get("injection_queue"):
        print("\n=== SUMMARY ===")
        summary_node = create_summary_node(llm)
        summary_update = summary_node(state)
        state = merge_state_updates(state, summary_update)
        print(f"Messages now: {len(state['messages'])}")
        print_last_message(state, "Summary response")

    print("\n=== EXECUTION FINISHED ===")
    print(f"Final messages count: {len(state.get('messages', []))}")
    print(f"Final thoughts count: {len(state.get('thoughts', []))}")
    print(f"Active agent: {state.get('active_agent')}")


# Quick entry point
if __name__ == "__main__":
    config = AppConfig.default_for_development()
    manual_graph_trace(config, use_real_llm=True)