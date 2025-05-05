import logging
from typing import Dict, Any, List
import json
from langgraph.graph import StateGraph, START, END
from .state import AgentState
from common.database import get_schema_description, execute_query
from common.query_processor import generate_sql_from_nl, check_sql_syntax, attempt_fix_sql
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from common.llm_factory import get_llm

logger = logging.getLogger(__name__)
MAX_FIX_ATTEMPTS = 1

# --- Helper to get latest user query ---
def get_last_user_query(state: AgentState) -> str | None:
    """Extracts the content of the last HumanMessage."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None

# --- Message Parsing Helpers ---
def find_last_message_by_name(messages: List[Any], name: str):
    for msg in reversed(messages):
        if hasattr(msg, "name") and msg.name == name:
            return msg
    return None

def find_last_human_message(messages: List[Any]):
    for msg in reversed(messages):
        if msg.__class__.__name__ == "HumanMessage":
            return msg
    return None

class WorkflowManager:
    def __init__(self) -> None:
        pass

    async def get_schema_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Getting schema")
        try:
            schema = await get_schema_description()
            if not schema or "Error" in schema:
                raise ValueError(f"Failed to retrieve schema: {schema}")
            return {"messages": [AIMessage(content=schema, name="schema")]}
        except Exception as e:
            logger.exception("Error in get_schema_node")
            return {"messages": [AIMessage(content=f"Fatal error getting schema: {e}", name="error")]} 

    async def generate_query_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Generating SQL query")
        messages = state["messages"]
        schema_msg = find_last_message_by_name(messages, "schema")
        nl_query_msg = find_last_human_message(messages)
        if not schema_msg:
            return {"messages": [AIMessage(content="Schema not available for query generation.", name="error")]}
        if not nl_query_msg:
            return {"messages": [AIMessage(content="Could not extract user query from messages.", name="error")]}
        try:
            sql = await generate_sql_from_nl(nl_query_msg.content, schema_msg.content)
            return {"messages": [AIMessage(content=sql, name="sql_query")]}
        except Exception as e:
            logger.exception("Error generating SQL")
            return {"messages": [AIMessage(content=f"Error generating SQL: {e}", name="error")]} 

    async def check_query_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Checking SQL query")
        messages = state["messages"]
        sql_msg = find_last_message_by_name(messages, "sql_query")
        if not sql_msg:
            return {"messages": [AIMessage(content="No SQL query found in history to check.", name="error")]}
        try:
            is_valid, error = await check_sql_syntax(sql_msg.content)
            if is_valid:
                logger.info("SQL syntax is valid.")
                return {"messages": [AIMessage(content="Syntax check OK", name="check_result")]} 
            else:
                logger.warning(f"SQL syntax invalid: {error}")
                return {"messages": [AIMessage(content=error or "Invalid SQL syntax", name="error")]} 
        except Exception as e:
            logger.exception("Error checking SQL syntax")
            return {"messages": [AIMessage(content=f"Error checking SQL: {e}", name="error")]} 

    async def execute_query_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Executing SQL query")
        messages = state["messages"]
        sql_msg = find_last_message_by_name(messages, "sql_query")
        if not sql_msg:
            return {"messages": [AIMessage(content="No SQL query found in history to execute.", name="error")]}
        try:
            results = await execute_query(sql_msg.content)
            logger.info(f"Query executed. Results count: {len(results)}")
            results_str = json.dumps(results)
            return {"messages": [AIMessage(content=results_str, name="results")]} 
        except Exception as e:
            logger.exception("Error executing query")
            return {"messages": [AIMessage(content=f"Error executing query: {e}", name="error")]} 

    async def fix_query_node(self, state: AgentState) -> Dict[str, Any]:
        logger.warning("Node: Attempting to fix SQL query")
        messages = state["messages"]
        invalid_sql_msg = find_last_message_by_name(messages, "sql_query")
        error_msg = find_last_message_by_name(messages, "error")
        schema_msg = find_last_message_by_name(messages, "schema")
        original_nl_query_msg = find_last_human_message(messages)
        if not all([invalid_sql_msg, error_msg, schema_msg, original_nl_query_msg]):
            return {"messages": [AIMessage(content="Cannot fix query: Missing context.", name="error")]} 
        try:
            fixed_sql = await attempt_fix_sql(
                invalid_sql=invalid_sql_msg.content,
                error_message=error_msg.content,
                schema=schema_msg.content,
                original_nl_query=original_nl_query_msg.content
            )
            return {"messages": [AIMessage(content=fixed_sql, name="sql_query")]} 
        except Exception as e:
            logger.exception("Error attempting to fix SQL")
            return {"messages": [AIMessage(content=f"Failed to attempt fix: {e}", name="error")]} 

    async def format_answer_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Formatting final answer")
        messages = state["messages"]
        llm = get_llm()
        question_msg = find_last_human_message(messages)
        results_msg = find_last_message_by_name(messages, "results")
        error_msg = find_last_message_by_name(messages, "error")
        if not llm:
            return {"messages": [AIMessage(content="Error: LLM not available to format the final answer.")]} 
        if not question_msg:
            return {"messages": [AIMessage(content="Error: Could not find original question to formulate answer.")]} 
        final_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Jij bent Fred, een hartstochtelijke fan van de voetbalclub Feyenoord uit Rotterdam-Zuid. Jouw doel is om vragen over Feyenoord te beantwoorden. Neem de vraag en de resultaten uit de database om een kort en bondig antwoord op de vraag te formuleren. Als de query geen resultaten opleverde, wees daar dan eerlijk over en zeg dat in je antwoord. Ga nooit antwoorden verzinnen."),
            ("human", "Vraag: {question}\nResultaten: {results}")
        ])
        if error_msg and (not results_msg or messages.index(error_msg) > messages.index(results_msg)):
            final_answer_content = f"Sorry, ik kon geen antwoord vinden. De volgende fout trad op: {error_msg.content}"
        elif results_msg:
            try:
                results = json.loads(results_msg.content)
                if not results:
                    prompt = final_answer_prompt.format(question=question_msg.content, results="Geen resultaten gevonden.")
                    ai_response = await llm.ainvoke(prompt)
                    final_answer_content = ai_response.content
                else:
                    formatted_results = "\n".join([str(row) for row in results])
                    prompt = final_answer_prompt.format(question=question_msg.content, results=formatted_results)
                    ai_response = await llm.ainvoke(prompt)
                    final_answer_content = ai_response.content
            except Exception as e:
                logger.exception("Error formatting results in final answer")
                final_answer_content = f"Sorry, er ging iets mis bij het formuleren van het antwoord: {e}"
        else:
            final_answer_content = "Sorry, er is een onverwachte status opgetreden in de workflow."
        return {"messages": [AIMessage(content=final_answer_content)]}

    def should_fix_or_execute(self, state: AgentState) -> str:
        logger.info("Condition: Checking if query needs fixing or execution.")
        messages = state["messages"]
        last_msg = messages[-1] if messages else None
        if hasattr(last_msg, "name") and last_msg.name == "error":
            return "fix_query"
        elif hasattr(last_msg, "name") and last_msg.name == "check_result":
            return "execute_query"
        else:
            return "format_answer"

    def after_execution(self, state: AgentState) -> str:
        logger.info("Condition: Checking result after execution.")
        messages = state["messages"]
        last_msg = messages[-1] if messages else None
        if hasattr(last_msg, "name") and last_msg.name == "error":
            return "fix_query"
        elif hasattr(last_msg, "name") and last_msg.name == "results":
            return "format_answer"
        else:
            return "format_answer"

    def create_workflow(self) -> StateGraph:
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("get_schema", self.get_schema_node)
        graph.add_node("generate_query", self.generate_query_node)
        graph.add_node("check_query", self.check_query_node)
        graph.add_node("execute_query", self.execute_query_node)
        graph.add_node("fix_query", self.fix_query_node)
        graph.add_node("format_answer", self.format_answer_node) # New node

        # Define edges
        graph.add_edge(START, "get_schema")
        graph.add_edge("get_schema", "generate_query")
        graph.add_edge("generate_query", "check_query")

        # Conditional edge after checking syntax
        graph.add_conditional_edges(
            "check_query",
            self.should_fix_or_execute,
            {
                "fix_query": "fix_query",
                "execute_query": "execute_query",
                "format_answer": "format_answer" # Route errors directly to formatting
            }
        )

        # Edge after attempting a fix (always go back to check)
        graph.add_edge("fix_query", "check_query")

        # Conditional edge after executing the query
        graph.add_conditional_edges(
            "execute_query",
            self.after_execution,
            {
                "fix_query": "fix_query",
                "format_answer": "format_answer" # Route success or final errors to formatting
            }
        )

        # Final edge from formatting the answer to the end
        graph.add_edge("format_answer", END)

        return graph

    def compile_graph(self):
        logger.info("Compiling LangGraph workflow...")
        compiled_graph = self.create_workflow().compile()
        logger.info("LangGraph workflow compiled.")
        return compiled_graph
