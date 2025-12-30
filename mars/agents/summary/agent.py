from mars.agents.base import create_agent_node
from mars.agents.summary.prompts import SUMMARY_SYSTEM_PROMPT
from datetime import datetime


def create_summary_node(llm):
    """
    Creates the summary agent node (minimal version - no tools).
    """
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    system_content = SUMMARY_SYSTEM_PROMPT.format(current_date=current_date_str)
    
    return create_agent_node(
        llm=llm,
        system_prompt=system_content,
        agent_name="summary"
    )