from mars.config import AppConfig
from mars.infrastructure.llm import create_llm
from mars.core.state import BasicAgentState
from mars.agents.orchestrator.agent import create_orchestrator_node
from mars.agents.summary.agent import create_summary_node

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
        initial_state = BasicAgentState(messages=[HumanMessage(content=user_message)])
        
        orch_result = self.orchestrator(initial_state)
        messages = orch_result["messages"]
        last_msg = messages[-1]
        
        print("Orchestrator:", last_msg.content)
        
        # Force summary for testing
        print("\n→ Forcing Summary Agent for validation...")
        summary_node = create_summary_node(self.llm)
        summary_result = summary_node(BasicAgentState(messages=messages))
        final_msg = summary_result["messages"][-1]
        
        print("Summary Agent:", final_msg.content)
        return final_msg.content

    # def run_once(self, user_message: str) -> str:
    #     """
    #     Simple multi-agent flow: Orchestrator → optional Summary
    #     """
    #     initial_state = BasicAgentState(
    #         messages=[HumanMessage(content=user_message)]
    #     )
        
    #     # Step 1: Orchestrator thinks
    #     orch_result = self.orchestrator(initial_state)
    #     messages = orch_result["messages"]
    #     last_msg = messages[-1]
        
    #     print("Orchestrator:", last_msg.content)
        
    #     # Step 2: Check if orchestrator wants to handoff to summary
    #     if "summarize" in last_msg.content.lower() or "merge" in last_msg.content.lower():
    #         summary_state = BasicAgentState(messages=messages)
    #         summary_node = create_summary_node(self.llm)
    #         summary_result = summary_node(summary_state)
    #         final_msg = summary_result["messages"][-1]
    #         print("\nSummary Agent:", final_msg.content)
    #         return final_msg.content
    #     else:
    #         return last_msg.content