from mars.config import AppConfig
from mars.infrastructure.llm import create_llm
from mars.core.state import BasicAgentState
from mars.agents.orchestrator.agent import create_orchestrator_node
from langchain_core.messages import HumanMessage


class SimpleRunner:
    """
    Very simple runner for initial testing - only orchestrator.
    Will be replaced with LangGraph based runner later.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = create_llm(config)
        self.orchestrator = create_orchestrator_node(self.llm)

    def run_once(self, user_message: str) -> str:
        """
        Simplest possible execution: just ask orchestrator one question
        """
        initial_state = BasicAgentState(
            messages=[HumanMessage(content=user_message)]
        )
        
        result = self.orchestrator(initial_state)
        last_message = result["messages"][-1]
        
        return last_message.content