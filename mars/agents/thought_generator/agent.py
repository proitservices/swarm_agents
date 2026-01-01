"""
Thought Generator Agent node factory
Generates new thoughts through a sequence of guided prompts
File location: ./mars/agents/thought_generator/agent.py
"""

# imports
from typing import Dict, List
from datetime import datetime

from langchain_core.messages import HumanMessage

from mars.agents.base import create_agent_node
from mars.agents.thought_generator.prompts import (
    THOUGHT_GENERATOR_SYSTEM_PROMPT,
    GUIDED_PROMPT_SEQUENCE
)
from mars.types import Thought, create_thought
from mars.infrastructure.logging import get_thought_generator_logger

# ──────────────────────────────────────────────────────────────────────────────
# Configurable constants
# ──────────────────────────────────────────────────────────────────────────────

MAX_GUIDED_STEPS = 4          # Safety limit to prevent long-running / freezing generation
DEFAULT_RELEVANCE_NEW = 0.68  # Default relevance score for newly generated thoughts

# ──────────────────────────────────────────────────────────────────────────────
# Core factory
# ──────────────────────────────────────────────────────────────────────────────

def create_thought_generator_node(llm, max_steps: int = MAX_GUIDED_STEPS):
    """
    Creates a Thought Generator node that produces new atomic Thoughts.

    Uses a controlled sequence of guided prompts to reflect on the current context
    and generate diverse new thoughts, grounded in prime seeds when available.

    Args:
        llm: The language model instance to use for generation
        max_steps: Maximum number of guided prompt steps to execute (default: 4)

    Returns:
        Callable: A node function that takes state and returns delta updates
    """
    current_date_str = datetime.utcnow().strftime("%Y-%m-%d")
    system_content = THOUGHT_GENERATOR_SYSTEM_PROMPT.format(current_date=current_date_str)

    base_node = create_agent_node(
        llm=llm,
        system_prompt=system_content,
        agent_name="thought_generator"
    )

    logger = get_thought_generator_logger()

    def thought_node(state: Dict) -> Dict:
        """
        Main generation logic with step limitation and early exit protection.

        Args:
            state: Current shared MARS state dictionary

        Returns:
            dict: Delta containing:
                - messages: New messages added during generation
                - thoughts: List of newly created Thought dictionaries
        """
        new_messages = []
        new_thoughts: List[Thought] = []

        # Early exit if no meaningful context exists
        if not state.get("messages") and not state.get("core_context"):
            logger.append("No meaningful context available → generation skipped")
            return {"messages": [], "thoughts": []}

        # Extract core context safely
        core_content = (
            state.get("core_context", {}).get("current_topic", "")
            or (state["messages"][-1].content if state.get("messages") else "")
        )

        # Gather prime thoughts for inspiration
        primes = [t for t in state.get("thoughts", []) if t["thought_id"].startswith("prime-")]
        prime_context = "\n".join(
            f"- {p['meta_narrative']}: {p['narrative'][:140]}..."
            for p in primes
        ) or "No prime thoughts loaded."

        # Limit the prompt sequence for safety/performance
        active_sequence = GUIDED_PROMPT_SEQUENCE[:max_steps]

        # Log start of generation cycle
        logger.append(
            f"Starting generation cycle — using {len(active_sequence)}/{len(GUIDED_PROMPT_SEQUENCE)} guided steps"
        )

        for i, prompt_template in enumerate(active_sequence, 1):
            # Prepare the full prompt for this step
            full_prompt = prompt_template.format(
                core_content=core_content,
                prime_context=prime_context
            )

            # Prepare input for LLM call
            input_state = state.copy()
            input_state["messages"] = input_state.get("messages", []) + [HumanMessage(content=full_prompt)]

            try:
                result = base_node(input_state)

                # Extract response safely
                response = result["messages"][-1].content if result.get("messages") else ""

                # Log this generation step
                logger.log_step(
                    step=i,
                    prompt=full_prompt,
                    reply=response
                )

                # Parse response into narrative + meta (fallback if format not followed)
                if "Meta:" in response:
                    narrative, meta_part = response.split("Meta:", 1)
                    meta_narrative = meta_part.strip()
                else:
                    narrative = response.strip()
                    meta_narrative = "Generated from guided reflection step"

                # Create new atomic Thought
                new_thought = create_thought(
                    narrative=narrative,
                    meta_narrative=meta_narrative,
                    origins=[p["thought_id"] for p in primes] if primes else [],
                    initial_relevance=DEFAULT_RELEVANCE_NEW,
                    is_seed=False
                )

                new_thoughts.append(new_thought)
                new_messages.extend(result.get("messages", []))

            except Exception as e:
                logger.append(f"WARNING: Generation step {i} failed → {str(e)}")
                break  # Stop on first serious error

        return {
            "messages": new_messages,
            "thoughts": new_thoughts
        }

    return thought_node