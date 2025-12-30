"""
Memory Swarm coordinator
Manages sequential evaluation of all active memory agents (simulating parallelism)
"""

# imports
from typing import Dict, Any, List

from mars.types import MARSState, Thought
from mars.agents.memory.single import create_single_memory_node


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
        - Builds injection queue and updated thoughts list

        Returns:
            dict: Only the keys that changed (for safe merging in LangGraph or diagnostic runner)
        """
        # Defensive initialization
        thoughts = state.get("thoughts", [])
        if not thoughts:
            return {"injection_queue": [], "active_agent": "memory_swarm"}

        new_queue: List[Thought] = []
        updated_thoughts = thoughts.copy()  # shallow copy - we replace objects

        for idx, thought in enumerate(thoughts):
            # Create and run specialized memory agent for this thought
            memory_node = create_single_memory_node(llm, thought)
            memory_result = memory_node(state)

            # Get latest message (evaluation result)
            if not memory_result.get("messages"):
                continue  # skip broken evaluations

            response_text = memory_result["messages"][-1].content.lower()

            # Current mock scoring logic (as requested - keep 0.85 for applicable)
            is_applicable = "applicable" in response_text or "yes" in response_text
            relevance_score = 0.85 if is_applicable else 0.42

            # Create updated thought version
            updated = thought.copy()
            updated["relevance_score"] = relevance_score
            updated["last_evaluated"] = "now"  # TODO: real timestamp later

            # Decision: injection or re-framing
            if relevance_score > 0.7:
                new_queue.append(updated)
            else:
                # Minimal re-framing for now
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