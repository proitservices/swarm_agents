"""
Single Memory Agent factory
Creates a specialized agent node for evaluating one specific Thought
"""

# imports
import json
from typing import Callable
from datetime import datetime

from mars.types import Thought, MARSState
from mars.agents.base import create_agent_node
from mars.agents.memory.prompts import MEMORY_SYSTEM_PROMPT


# helper classes
# (none needed - responsibility delegated to base agent node)


# operational classes
# (none needed here)


# Core factory
def create_single_memory_node(
    llm,
    thought: Thought
) -> Callable[[MARSState], dict]:
    """
    Factory that creates a memory agent node specialized for one Thought.

    Args:
        llm: Language model instance (real or mock)
        thought: The specific Thought this agent is responsible for

    Returns:
        Callable[[MARSState], dict]: Node function returning delta updates
    """
    current_date_str = datetime.now().strftime("%Y-%m-%d")

    # 1. Serialize thought to JSON
    # 2. Escape curly braces to prevent LangChain prompt variable parsing
    thought_json = json.dumps(thought, indent=2)
    thought_safe = thought_json.replace("{", "{{").replace("}", "}}")

    # Build system prompt with safely escaped thought content
    system_content = (
        MEMORY_SYSTEM_PROMPT.format(current_date=current_date_str)
        + "\n\nHeld Thought to evaluate (as JSON):\n"
        + thought_safe
    )

    # Create base agent node (prompt | llm chain + invocation logic)
    base_node = create_agent_node(
        llm=llm,
        system_prompt=system_content,
        agent_name=f"memory_{thought['thought_id']}"
    )

    def memory_node(state: MARSState) -> dict:
        """
        Executes base node and returns only the delta (new messages).
        Compatible with both LangGraph and manual state merging.
        """
        full_update = base_node(state)
        return {
            "messages": full_update.get("messages", [])
        }

    return memory_node