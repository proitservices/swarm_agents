from langchain_openai import ChatOpenAI
from mars.config import AppConfig


def create_llm(config: AppConfig) -> ChatOpenAI:
    """
    Factory function that creates configured LLM instance.
    
    Returns:
        ChatOpenAI: ready-to-use chat model with tool calling support
    """
    return ChatOpenAI(
        base_url=config.llm.base_url,
        api_key=config.llm.api_key.get_secret_value(),
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        # Important for Mistral-like models via vLLM/OpenAI compat API
        extra_body={"model": config.llm.model},  # sometimes needed
    )