# # mars/infrastructure/llm.py
# from langchain_openai import ChatOpenAI
# from mars.config import AppConfig


# def create_llm(config: AppConfig) -> ChatOpenAI:
#     """
#     Creates a ChatOpenAI instance that works reliably with Mistral served via vLLM.
#     No extra parameters needed — most compatibility issues are handled by the endpoint.
#     """
#     return ChatOpenAI(
#         base_url=config.llm.base_url,
#         api_key=config.llm.api_key.get_secret_value(),
#         model=config.llm.model,
#         temperature=config.llm.temperature,
#         max_tokens=config.llm.max_tokens,
#         # Critical fixes for vLLM hang
#         streaming=False,                     # Explicitly disable streaming
#         request_timeout=60,                  # Hard timeout per request (seconds)
#         max_retries=0,                       # No auto-retries (debug faster)
#         http_client_kwargs={"timeout": 60}   # Extra safety for underlying httpx
#     )


# ./mars/infrastructure/llm.py
"""
LLM factory with clean, compatible configuration for vLLM + Mistral endpoint.
Removes deprecated http_client_kwargs and uses safe defaults.

File location: ./mars/infrastructure/llm.py
"""

# imports
from langchain_openai import ChatOpenAI
from mars.config import AppConfig


def create_llm(config: AppConfig) -> ChatOpenAI:
    """
    Creates a ChatOpenAI instance optimized for vLLM/Mistral compatibility.

    Args:
        config (AppConfig): Application configuration with LLM settings

    Returns:
        ChatOpenAI: Configured model instance
    """
    return ChatOpenAI(
        base_url=config.llm.base_url,
        api_key=config.llm.api_key.get_secret_value(),
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        # Explicitly disable streaming (prevents hang in sync invoke)
        streaming=False,
        # Hard timeouts to prevent silent hangs
        request_timeout=120,
        # No retries during debug
        max_retries=0,
        # No deprecated kwargs
        # http_client_kwargs removed - was causing error in older langchain-openai
    )
