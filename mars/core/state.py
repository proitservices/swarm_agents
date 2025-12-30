from typing import TypedDict, Annotated, List
from langchain_core.messages import AnyMessage
import operator


class BasicAgentState(TypedDict):
    """Minimal state used in first iteration (only messages)"""
    messages: Annotated[List[AnyMessage], operator.add]