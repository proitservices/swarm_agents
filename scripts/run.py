#!/usr/bin/env python3
import sys
from mars.config import AppConfig
from mars.core.runner import SimpleRunner


def main():
    config = AppConfig.default_for_development()
    runner = SimpleRunner(config)

    # Test cases – choose one at a time
    test_messages = [
        "Please summarize the key principles of tort law in New York.",
        "List the main elements of negligence in New York tort law and then summarize them concisely.",
        "Explain duty of care in NY tort law. After that please summarize your answer."
    ]

    for msg in test_messages:
        print("\n" + "="*60)
        print("User:", msg)
        print("-" * 50)
        
        answer = runner.run_once(msg)
        print("\nFinal Answer:", answer)
        print("="*60)


if __name__ == "__main__":
    main()