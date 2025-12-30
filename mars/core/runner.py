# ./mars/core/runner.py
"""
High-level runner facade for the MARS swarm graph.
Initializes LLM, graph and executes workflows with proper checkpointing.
Handles thread management for persistence and state resumption.
"""

# imports
from typing import List

from langchain_core.messages import AnyMessage, HumanMessage

from mars.config import AppConfig
from mars.infrastructure.llm import create_llm
from mars.core.graph import MARSGraph
from mars.core.state import MARSState


class MARSRunner:
    """
    Core runner class that ties configuration, LLM and graph together.
    Provides simple run interface while managing thread_id for checkpointer.
    """

    def __init__(self, config: AppConfig):
        """
        Initializes the runner with configuration.

        Args:
            config (AppConfig): Application-wide configuration object
        """
        self.config = config
        self.llm = create_llm(config)
        self.graph = MARSGraph(self.llm)
        self.thread_id = "mars-thread-001"  # Fixed thread ID for prototype

    def run(self, user_message: str) -> str:
        """
        Executes the full MARS swarm workflow for a single user query.

        Args:
            user_message (str): User input question/query

        Returns:
            str: Final answer extracted from the last message
        """
        # Prepare initial state
        initial_state: MARSState = {
            "messages": [HumanMessage(content=user_message)],
            "core_context": {},
            "thoughts": [],
            "injection_queue": [],
            "active_agent": "orchestrator",
            "last_handoff_reason": None
        }

        # Prepare LangGraph config with required thread_id
        graph_config = {
            "configurable": {
                "thread_id": self.thread_id,
            }
        }

        # Execute the graph - pass config correctly to the underlying app
        # Since MARSGraph.invoke() is a wrapper, we must call the internal app with config
        final_state = self.graph.app.invoke(
            input=initial_state,
            config=graph_config
        )

        # Extract final answer from messages
        final_messages: List[AnyMessage] = final_state.get("messages", [])
        if final_messages:
            return final_messages[-1].content

        return "No final answer generated (empty message history)."


# Core execution block (when file is run directly)
if __name__ == "__main__":
    config = AppConfig.default_for_development()
    runner = MARSRunner(config)

    test_query = "Summarize key principles of tort law in New York."
    print("User:", test_query)
    print("-" * 50)

    answer = runner.run(test_query)
    print("Final Answer:")
    print(answer)