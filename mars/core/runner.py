# mars/core/runner.py
"""
Simple runner for early development stage - orchestrator + summary
"""

# imports
from mars.config import AppConfig
from mars.infrastructure.llm import create_llm
from mars.core.state import BasicAgentState
from mars.agents.orchestrator.agent import create_orchestrator_node
from mars.agents.summary.agent import create_summary_node
from langchain_core.messages import HumanMessage


class SimpleRunner:
    """Minimal runner to test multi-agent coordination"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = create_llm(config)
        self.orchestrator = create_orchestrator_node(self.llm)

    def run_once(self, user_message: str) -> str:
        """
        Orchestrator → optional summary → final answer from orchestrator
        """
        initial_state = BasicAgentState(messages=[HumanMessage(content=user_message)])

        # ── Orchestrator ─────────────────────────────────────────────────────
        print("=== orchestrator ===")
        orch_result = self.orchestrator(initial_state)
        messages = orch_result["messages"]
        orch_content = messages[-1].content
        print(orch_content)
        print("=== end of orchestrator ===\n")

        # Decide whether to call summary (simple keyword check for now)
        should_summarize = any(
            word in orch_content.lower()
            for word in ["summarize", "condense", "short", "summary"]
        )

        if should_summarize:
            print("→ Calling Summary Agent...\n")
            print("=== summary ===")
            summary_node = create_summary_node(self.llm)
            summary_result = summary_node(BasicAgentState(messages=messages))
            summary_content = summary_result["messages"][-1].content
            print(summary_content)
            print("=== end of summary ===\n")
        else:
            summary_content = None

        # ── Final answer always comes from orchestrator ─────────────────────
        # We simulate this by asking orchestrator to give final clean version
        final_prompt = (
            "You are the Orchestrator. Provide the clean final answer "
            "based on previous reasoning.\n\n"
            f"Previous content:\n{orch_content}"
        )
        if summary_content:
            final_prompt += f"\nSummary version:\n{summary_content}"

        final_state = BasicAgentState(messages=[HumanMessage(content=final_prompt)])
        final_result = self.orchestrator(final_state)
        final_answer = final_result["messages"][-1].content

        print("Final Answer (produced by orchestrator):")
        print(final_answer)

        return final_answer