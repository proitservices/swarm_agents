# Quick standalone test - put in scripts/llm_smoke_test.py
from mars.infrastructure.llm import create_llm
from mars.config import AppConfig

config = AppConfig.default_for_development()
llm = create_llm(config)

messages = [{"role": "user", "content": "Say 'MARS swarm is alive' if you receive this."}]
response = llm.invoke(messages)
print(response.content)