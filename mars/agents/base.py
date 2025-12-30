from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from typing import Callable, Dict, Any


def create_agent_node(
    llm: BaseChatModel,
    system_prompt: str,
    agent_name: str
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Factory for simple agent nodes (no tools yet).
    
    Args:
        llm: configured language model
        system_prompt: system message content
        agent_name: identifier for logging/state
    
    Returns:
        Callable that takes state and returns state update
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}")
    ])
    
    chain = prompt | llm

    def node(state: Dict) -> Dict:
        response = chain.invoke({"messages": state["messages"]})
        return {
            "messages": [response],
            # We record which agent produced last message
            "last_agent": agent_name
        }
    
    node.__name__ = f"{agent_name}_node"
    return node