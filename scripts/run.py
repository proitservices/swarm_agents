# # ./scripts/run.py
# """
# Main entry point script for running the MARS swarm prototype.
# Handles configuration loading, runner/graph instantiation and simple execution.
# Follows clean separation: only orchestrates, delegates to core components.
# """

# # imports
# from mars.config import AppConfig
# from mars.core.diagnostic_runner import run_diagnostic
# from mars.core.state import MARSState
# from langchain_core.messages import HumanMessage


# def main():
#     """
#     Main execution flow.
#     Loads config, creates runner, runs a simple test query.
#     """
#     # Configuration
#     config = AppConfig.default_for_development()

#     # Runner (preferred way - encapsulates graph + LLM)
#     runner = DiagnosticMARSRunner(config)

#     # Test query
#     test_query = "Summarize key principles of tort law in New York."

#     print("User:", test_query)
#     print("-" * 50)

#     # Execute
#     final_answer = runner.run_with_timeout(test_query)

#     print("Final Answer:")
#     print(final_answer)


# if __name__ == "__main__":
#     main()


# ./scripts/run.py
from mars.config import AppConfig
from mars.core.diagnostic_runner import manual_graph_trace

def main():
    config = AppConfig.default_for_development()
    manual_graph_trace(config)

if __name__ == "__main__":
    main()