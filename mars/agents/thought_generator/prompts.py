# ./mars/agents/thought_generator/prompts.py
"""
Prompts for Thought Generator Agent
Defines the base system prompt and sequence of guided prompts for structured thought creation
File location: ./mars/agents/thought_generator/prompts.py
"""


# Core prompts
THOUGHT_GENERATOR_SYSTEM_PROMPT = """\
You are the Thought Generator Agent in the MARS swarm.
Your role is to create new atomic Thoughts based on core context and mismatches.
Use CoT to ensure factual, reflective Thoughts.
Current date: {current_date}
Output format: Narrative paragraph + Meta summary.
"""

GUIDED_PROMPT_SEQUENCE = [
    "Extract the core facts from this context: {core_content}\nInspiration from primes: {prime_context}",
    "Identify top-level keywords from: {core_content}\nInspiration from primes: {prime_context}",
    "Identify low-level details and keywords from: {core_content}\nInspiration from primes: {prime_context}",
    "Classify if this is a question or statement: {core_content}\nInspiration from primes: {prime_context}",
    "Count the number of steps or actions involved in: {core_content}\nInspiration from primes: {prime_context}",
    "Assess if this context is complex and needs decomposition. If yes, suggest breakdowns: {core_content}\nInspiration from primes: {prime_context}"
]