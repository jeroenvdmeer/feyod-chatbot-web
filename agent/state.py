from typing import TypedDict, Annotated, List, Dict, Any, Optional

from langgraph.graph.message import add_messages

class AgentState(TypedDict, total=False):
    # Core conversation history, managed by the reducer
    messages: Annotated[list, add_messages]

    # Data passed between nodes
    schema: Optional[str]
    sql_query: Optional[str]
    error_message: Optional[str]
    results: Optional[List[Dict[str, Any]]]
    fix_attempts: int # Use int, initialize in get_schema_node
