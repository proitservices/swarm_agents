# class description
"""
Factory module for creating agent nodes in the swarm.
Each node is a callable that takes current state and returns state update.
Handles proper message formatting for Mistral/vLLM endpoints.
"""

# imports
from typing import Callable, Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage


def create_agent_node(
    llm: BaseChatModel,
    system_prompt: str,
    agent_name: str
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Creates a node function for an agent.

    Args:
        llm: configured language model instance
        system_prompt: content of the persistent system message
        agent_name: identifier used for logging and state tracking

    Returns:
        Callable[[Dict[str, Any]], Dict[str, Any]]: node function that updates state
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}")
    ])

    chain = prompt | llm

    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        messages: List[AnyMessage] = state["messages"]

        invoke_messages = messages.copy()

        # When continuing from an assistant message we need to signal continuation
        # vLLM/Mistral often requires an empty user message in this situation
        if messages and isinstance(messages[-1], AIMessage):
            invoke_messages.append(HumanMessage(content=""))

        response = chain.invoke({"messages": invoke_messages})

        return {
            "messages": [response],
            "last_agent": agent_name
        }

    node.__name__ = f"{agent_name}_node"
    return node