"""
Orchestrator agent node factory
Now includes reasoning trace logging
"""

# imports
from datetime import datetime

from mars.agents.base import create_agent_node
from mars.agents.orchestrator.prompts import ORCHESTRATOR_SYSTEM_PROMPT
from mars.infrastructure.logging import get_orchestrator_logger


# helper classes (none needed here)

# operational classes (none needed here)

# Core factory
def create_orchestrator_node(llm):
    """
    Creates the orchestrator agent node with logging hook.
    Logs every generated response + context to orchestrator_dialogue.log
    """
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    system_content = ORCHESTRATOR_SYSTEM_PROMPT.format(current_date=current_date_str)

    base_node = create_agent_node(
        llm=llm,
        system_prompt=system_content,
        agent_name="orchestrator"
    )

    logger = get_orchestrator_logger()
    step_counter = 0

    def orchestrator_node(state: dict) -> dict:
        nonlocal step_counter
        step_counter += 1

        # Execute normal orchestrator logic
        update = base_node(state)

        # Extract latest message (the orchestrator's response)
        messages = update.get("messages", [])
        if messages:
            response_text = messages[-1].content

            # Log the step
            logger.log_step(
                step_number=step_counter,
                orchestrator_output=response_text,
                # You can pass injected_summary from state if you store it
                # injected_summary=state.get("last_injection", None),
                active_thought_ids=[t["thought_id"] for t in state.get("thoughts", [])]
            )

        return update

    return orchestrator_node