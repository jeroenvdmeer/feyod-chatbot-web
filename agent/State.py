from typing import TypedDict, Annotated, List
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages] # Core conversation history, managed by the reducer
    resolved_entities: dict[str, str]  # Track entity mappings throughout the workflow
    schema: str  # Cached database schema
    schema_timestamp: str  # ISO timestamp when schema was retrieved

def find_last_message_by_name(messages: List[AnyMessage], name: str) -> AnyMessage | None:
    """Find the last message with a specific name in the list of messages."""
    for msg in reversed(messages):
        if hasattr(msg, "name") and msg.name == name:
            return msg
    return None

def find_last_human_message(messages: List[AnyMessage]) -> HumanMessage | None:
    """Find the last HumanMessage in the list of messages."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    return None
