from pydantic import BaseModel, Field, SecretStr
from typing import Literal


class LLMConfig(BaseModel):
    """Configuration for LLM connection"""

    base_url: str = Field(..., description="OpenAI-compatible API endpoint")
    api_key: SecretStr = Field(..., description="API key for the endpoint")
    model: str = Field(..., description="Model identifier/path")
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int | None = None


class AppConfig(BaseModel):
    """Global application configuration"""

    llm: LLMConfig = Field(...)

    # Later we will add:
    # neo4j_uri: str
    # qdrant_url: str
    # etc.

    @classmethod
    def default_for_development(cls) -> "AppConfig":
        """Development / quick testing configuration"""
        return cls(
            llm=LLMConfig(
                base_url="https://yewsfcisrhdkhr-7861.proxy.runpod.net/v1",
                api_key=SecretStr("dummy-key"),
                model="/workspace/opt/models/mistral",
                temperature=0.0,
            )
        )