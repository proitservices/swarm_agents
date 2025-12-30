# mars/infrastructure/llm.py
from langchain_openai import ChatOpenAI
from mars.config import AppConfig


def create_llm(config: AppConfig) -> ChatOpenAI:
    """
    Creates a ChatOpenAI instance that works reliably with Mistral served via vLLM.
    No extra parameters needed — most compatibility issues are handled by the endpoint.
    """
    return ChatOpenAI(
        base_url=config.llm.base_url,
        api_key=config.llm.api_key.get_secret_value(),
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )