"""
Memory Swarm coordinator
Manages sequential evaluation of all active memory agents (simulating parallelism)
"""

# imports
from typing import Dict, Any, List

from mars.types import MARSState, Thought
from mars.agents.memory.single import create_single_memory_node
from mars.infrastructure.logging import get_thought_logger


# helper classes
# (none required at swarm level - single agents handle their own logic)


# operational classes
# (thought generator imported lazily to avoid circular imports)


# Core factory
def create_memory_swarm_node(llm):
    """
    Creates the memory swarm node - evaluates all existing thoughts.

    Args:
        llm: Language model instance used by individual memory agents

    Returns:
        Callable[[MARSState], dict]: Node function returning only delta updates
    """
    def swarm_node(state: MARSState) -> Dict[str, Any]:
        """
        Main swarm processing logic:
        - Loops through all current thoughts
        - Runs dedicated memory agent for each
        - Applies simple mock scoring (0.85 for applicable)
        - Logs evaluation decision per thought
        - Builds injection queue and updated thoughts list

        Returns:
            dict: Only the keys that changed (for safe merging)
        """
        # Defensive initialization
        thoughts = state.get("thoughts", [])
        if not thoughts:
            return {"injection_queue": [], "active_agent": "memory_swarm"}

        new_queue: List[Thought] = []
        updated_thoughts = thoughts.copy()  # shallow copy - we replace objects

        # Extract current orchestrator context once (same for all thoughts in this cycle)
        last_msg = state["messages"][-1].content
        context_snippet = (
            last_msg[:400] + "..." if len(last_msg) > 400 else last_msg
        )

        for idx, thought in enumerate(thoughts):
            # Create and run specialized memory agent for this thought
            memory_node = create_single_memory_node(llm, thought)
            memory_result = memory_node(state)

            # Get latest evaluation message
            if not memory_result.get("messages"):
                continue  # skip broken evaluations

            response_text = memory_result["messages"][-1].content.lower()

            # Mock scoring (kept as requested)
            is_applicable = "applicable" in response_text or "yes" in response_text
            relevance_score = 0.85 if is_applicable else 0.42

            # Log the memory agent's assessment
            thought_logger = get_thought_logger(thought)
            decision = "inject" if relevance_score > 0.7 else "reframe"

            # Upgrade decision if mismatch detected
            if "mismatch" in response_text or "conflict" in response_text:
                decision = "generate_new" if decision == "reframe" else decision

            thought_logger.log_evaluation(
                current_orchestrator_snippet=context_snippet,
                relevance_score=relevance_score,
                decision=decision,
                reasoning=f"Evaluation response: {response_text[:180]}..."
            )

            # Create updated thought version
            updated = thought.copy()
            updated["relevance_score"] = relevance_score
            updated["last_evaluated"] = "now"  # TODO: real timestamp later

            # Apply decision
            if relevance_score > 0.7:
                new_queue.append(updated)
            else:
                # Minimal re-framing
                updated["narrative"] = updated.get("narrative", "") + " [re-framed]"

                # Optional: mismatch → new thought generation
                if "mismatch" in response_text or "conflict" in response_text:
                    try:
                        from mars.agents.thought_generator.agent import create_thought_generator_node
                        thought_gen = create_thought_generator_node(llm)
                        gen_result = thought_gen(state)
                        if "thoughts" in gen_result and gen_result["thoughts"]:
                            new_thought = gen_result["thoughts"][-1]
                            updated_thoughts.append(new_thought)
                    except Exception as gen_error:
                        print(f"Warning: New thought generation failed: {gen_error}")

                updated_thoughts[idx] = updated  # replace in place

        return {
            "thoughts": updated_thoughts,
            "injection_queue": new_queue,
            "active_agent": "memory_swarm"
        }

    return swarm_node