"""
Thought Generator Agent node.
"""

from mars.agents.base import create_agent_node
from mars.agents.thought_generator.prompts import THOUGHT_GENERATOR_SYSTEM_PROMPT
from mars.types import Thought
from datetime import datetime

def create_thought_generator_node(llm):
    """
    Creates Thought Generator node.

    Args:
        llm: Configured LLM.

    Returns:
        Callable: Node function.
    """
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    system_content = THOUGHT_GENERATOR_SYSTEM_PROMPT.format(current_date=current_date_str)
    
    node_func = create_agent_node(llm, system_content, "thought_generator")

    def thought_node(state: dict) -> dict:
        """
        Wraps base node to produce Thought dict from response.

        Args:
            state (dict): Current state.

        Returns:
            dict: Updated state with new Thought.
        """
        result = node_func(state)
        response = result["messages"][-1].content
        # Parse response into Narrative and Meta (simple split for now)
        parts = response.split("Meta:") if "Meta:" in response else [response, ""]
        new_thought = create_thought(parts[0].strip(), parts[1].strip() if len(parts) > 1 else "")
        state["thoughts"].append(new_thought)
        return result | {"thoughts": state["thoughts"]}

    return thought_node