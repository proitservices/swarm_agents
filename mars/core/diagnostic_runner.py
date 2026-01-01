"""
Diagnostic Runner for MARS Swarm - Manual step-by-step execution
Loads prime seed thoughts from disk, expands them via Thought Generator,
then runs the main reasoning loop with orchestrator → swarm → summary.
Displays both prime and dynamically generated thoughts at each stage.
File location: ./mars/core/diagnostic_runner.py
"""

# imports
import time
import json
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.messages import AIMessage, HumanMessage, AnyMessage

from mars.types import MARSState, create_thought
from mars.infrastructure.llm import create_llm
from mars.agents.orchestrator.agent import create_orchestrator_node
from mars.agents.summary.agent import create_summary_node
from mars.agents.memory.swarm import create_memory_swarm_node
from mars.agents.thought_generator.agent import create_thought_generator_node
from mars.config import AppConfig

# ──────────────────────────────────────────────────────────────────────────────
# Helper classes
# ──────────────────────────────────────────────────────────────────────────────

class MockLLM:
    """Minimal mock fallback - not used when use_real_llm=True"""
    def invoke(self, input_data, **kwargs):
        return AIMessage(content=f"Mock response {time.strftime('%H:%M:%S')}")

# ──────────────────────────────────────────────────────────────────────────────
# Operational classes
# ──────────────────────────────────────────────────────────────────────────────

def load_seed_thoughts(seeds_dir: str = "mars/memories/seeds") -> List[Dict[str, Any]]:
    """
    Loads all prime seed thoughts from JSON files in the seeds directory.

    Returns:
        List[Dict[str, Any]]: List of loaded thought dictionaries
    """
    seeds_path = Path(seeds_dir)
    if not seeds_path.exists():
        print(f"Warning: Seeds directory not found: {seeds_path}")
        return []

    loaded_thoughts = []
    for file_path in seeds_path.glob("prime-thought-*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if all(key in data for key in ["thought_id", "narrative", "meta_narrative"]):
                    loaded_thoughts.append(data)
                else:
                    print(f"Warning: Invalid seed file format: {file_path}")
        except Exception as e:
            print(f"Error loading seed file {file_path}: {e}")

    print(f"Loaded {len(loaded_thoughts)} seed thoughts from {seeds_path}")
    return loaded_thoughts

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def merge_state_updates(original: MARSState, update: Dict[str, Any]) -> MARSState:
    """Safely merges node updates into full state without losing keys."""
    new_state = original.copy()
    for key, value in update.items():
        if isinstance(value, list) and key in ("messages", "thoughts", "injection_queue"):
            new_state[key] = new_state.get(key, []) + value
        else:
            new_state[key] = value
    return new_state


def print_last_message(state: MARSState, label: str = "Last message") -> None:
    """Prints the content of the most recent message with a label."""
    messages = state.get("messages", [])
    if not messages:
        print(f"{label}: (no messages)")
        return
    last_msg = messages[-1]
    content = getattr(last_msg, 'content', str(last_msg)).strip()
    if len(content) > 280:
        content = content[:280] + "..."
    print(f"{label}:\n{content}\n")


def print_thought_summary(thoughts: List[Dict[str, Any]], prefix: str = "Thoughts") -> None:
    """
    Prints a concise summary of all current thoughts (prime + generated).
    """
    print(f"{prefix}: {len(thoughts)}")
    for t in thoughts:
        tid = t["thought_id"]
        meta = t.get("meta_narrative", "no meta")
        score = t.get("relevance_score", 0.0)
        prefix_str = " prime" if tid.startswith("prime-") else " gen "
        print(f"{prefix_str} - {tid} | relevance: {score:.2f} | {meta[:70]}...")


# ──────────────────────────────────────────────────────────────────────────────
# Core execution logic
# ──────────────────────────────────────────────────────────────────────────────

def manual_graph_trace(
    config: AppConfig,
    use_real_llm: bool = True,
    query: str = "Summarize key principles of tort law in New York. Then propose a new law after careful consideration and deep thinking. When the new law is formed assess its application to a example case with your reasoning presented in detail."
) -> None:
    """
    Runs manual step-by-step execution of the MARS swarm graph.
    1. Loads prime seeds
    2. Uses Thought Generator to create initial expanded thought set
    3. Runs orchestrator → swarm → summary loop
    Shows evolution of thoughts at each stage.
    """
    print("=== MANUAL GRAPH TRACE ===")
    print(f"LLM: {'REAL vLLM' if use_real_llm else 'MOCK'}")
    print("Query:", query)
    print("Start:", time.strftime("%H:%M:%S"))

    llm = create_llm(config) if use_real_llm else MockLLM()

    # Step 1: Load prime seeds from disk
    seed_thoughts = load_seed_thoughts()

    # Fallback if no seeds found
    if not seed_thoughts:
        fallback = create_thought(
            narrative=f"Initial parsing of query: {query}",
            meta_narrative="Diagnostic fallback seed · Legal domain",
            initial_relevance=0.65
        )
        seed_thoughts = [fallback]

    # Step 2: Use Thought Generator to expand the initial thought space
    print("\n=== INITIAL THOUGHT GENERATION (seed spawning + guided reflection) ===")
    thought_gen = create_thought_generator_node(llm)
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "thoughts": seed_thoughts,
        "core_context": {"topic": "Initial legal reasoning setup"}
    }
    
    gen_delta = thought_gen(initial_state)
    generated_thoughts = gen_delta.get("thoughts", [])
    
    # Combine primes + generated thoughts (deduplication happens later in swarm)
    all_initial_thoughts = seed_thoughts + [t for t in generated_thoughts]
    
    print(f"→ Generated {len(generated_thoughts)} new thoughts")
    print_thought_summary(all_initial_thoughts, "After initial generation")

    # Step 3: Initialize full state with expanded thoughts
    state: MARSState = {
        "messages": [HumanMessage(content=query)],
        "core_context": {"topic": "Principles of tort law in New York"},
        "thoughts": all_initial_thoughts,
        "injection_queue": [],
        "active_agent": "orchestrator",
        "last_handoff_reason": None
    }

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
    print_thought_summary(state.get("thoughts", []), "After swarm")
    print(f"Injection queue size: {len(state.get('injection_queue', []))}")
    if state.get("injection_queue"):
        print("Queued thought IDs:", [t["thought_id"] for t in state["injection_queue"]])

    # ─────────────── Summary ───────────────
    if state.get("injection_queue"):
        print("\n=== SUMMARY ===")
        summary_node = create_summary_node(llm)
        summary_update = summary_node(state)
        state = merge_state_updates(state, summary_update)
        print(f"Messages now: {len(state['messages'])}")
        print_last_message(state, "Summary response")

    print("\n=== EXECUTION FINISHED ===")
    print(f"Final messages count: {len(state.get('messages', []))}")
    print_thought_summary(state.get("thoughts", []), "Final thoughts")
    print(f"Active agent: {state.get('active_agent')}")


# Quick entry point
if __name__ == "__main__":
    config = AppConfig.default_for_development()
    manual_graph_trace(config, use_real_llm=True)