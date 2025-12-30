#!/usr/bin/env python3
import sys
from mars.config import AppConfig
from mars.core.runner import SimpleRunner


def main():
    config = AppConfig.default_for_development()
    runner = SimpleRunner(config)

    test_message = "Hello! Please tell me what is the current date and what is your role."
    
    print("User:", test_message)
    print("-" * 50)
    
    answer = runner.run_once(test_message)
    print("Orchestrator:", answer)


if __name__ == "__main__":
    main()