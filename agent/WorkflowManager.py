import logging
import json
from datetime import datetime
import re
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from .State import AgentState, find_last_message_by_name, find_last_human_message
from common.database import get_schema_description, execute_query
from common.query_processor import generate_sql_from_nl, check_sql_syntax, attempt_fix_sql
from common.utils.entity_resolution import find_ambiguous_entities, resolve_entities
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from common.llm_factory import get_llm
from common.utils.memory_utils import _prepare_llm_context

logger = logging.getLogger(__name__)
MAX_FIX_ATTEMPTS = 1

def log_message_contents(messages, logger, prefix=""):
    for i, msg in enumerate(messages):
        msg_type = getattr(msg, "__class__", type(msg)).__name__
        name = getattr(msg, "name", None)
        content = getattr(msg, "content", None)
        logger.info(f"{prefix}Message[{i}]: type={msg_type}, name={name}, content={content}")

class WorkflowManager:
    def __init__(self) -> None:
        pass

    def canonicalize_query(self, user_query: str, resolved_entities: dict) -> str:
        """Replace all user mentions in the query with their canonical names."""
        # Clean up input query - normalize whitespace and line endings
        result = ' '.join(user_query.split())
        
        for mention, canonical in resolved_entities.items():
            # Clean up mention - normalize whitespace
            clean_mention = ' '.join(mention.split())
            # Create pattern that matches the whole word/phrase
            pattern = r'\b' + re.escape(clean_mention) + r'\b'
            # Replace all case-insensitive matches
            result = re.sub(pattern, canonical, result, flags=re.IGNORECASE)
        
        logger.info(f"Canonicalized query: '{result}'")
        return result

    async def get_schema_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Getting schema")
        log_message_contents(state.get("messages", []), logger, prefix="get_schema_node-")
        schema = state.get("schema")
        schema_timestamp = state.get("schema_timestamp")
        if not schema:
            try:
                schema = await get_schema_description()
                if not schema or "Error" in schema:
                    raise ValueError(f"Failed to retrieve schema: {schema}")
                logger.info("Schema fetched and cached in state.")
                schema_timestamp = datetime.utcnow().isoformat()
            except Exception as e:
                logger.exception("Error in get_schema_node")
                return {"messages": [AIMessage(content=f"Fatal error getting schema: {e}", name="error")]}
        else:
            logger.info(f"Using cached schema from {state.get("schema_timestamp")}")

        return {
            "messages": [AIMessage(content="Schema loaded.", name="schema")],
            "schema": schema,
            "schema_timestamp": schema_timestamp
        }

    async def clarify_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Clarifying ambiguous entities")
        log_message_contents(state.get("messages", []), logger, prefix="clarify_node-")
        messages = state["messages"]
        nl_query_msg = find_last_human_message(messages)
        if not nl_query_msg:
            return {"messages": [AIMessage(content="Kon de gebruikersvraag niet vinden voor clarificatie.", name="error")],
                    "resolved_entities": state.get("resolved_entities", {})}

        ambiguous_entities = await find_ambiguous_entities(nl_query_msg.content)
        new_resolved = await resolve_entities(nl_query_msg.content)
        resolved_entities = state.get("resolved_entities") or {}
        resolved_entities.update(new_resolved)
        state["resolved_entities"] = resolved_entities
        # --- End merge ---
        if ambiguous_entities:
            clarification_text = (
                f"Ik heb meerdere mogelijkheden gevonden voor: {', '.join(ambiguous_entities)}. "
                "Kun je specifieker zijn over welke speler, club of competitie je bedoelt?"
            )
            return {"messages": [AIMessage(content=clarification_text, name="clarify")],
                    "resolved_entities": resolved_entities}
        return {"messages": [AIMessage(content="OK", name="clarified")],
                "resolved_entities": resolved_entities}

    async def generate_query_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Generating SQL query")
        log_message_contents(state.get("messages", []), logger, prefix="generate_query_node-")
        messages = state["messages"]
        # Prepare context for LLM (if needed for future LLM-based query generation)
        context_messages = await _prepare_llm_context(messages)
        logger.info(f"Using prepared LLM context with {len(context_messages)} messages for query generation.")
        log_message_contents(context_messages, logger, prefix="generate_query_node-PreparedContext-")
        nl_query_msg = find_last_human_message(context_messages)

        schema = state.get("schema")
        if not schema:
            return {"messages": [AIMessage(content="Schema not available for query generation.", name="error")],
                    "resolved_entities": state.get("resolved_entities", {})}
        if not nl_query_msg:
            return {"messages": [AIMessage(content="Could not extract user query from messages.", name="error")],
                    "resolved_entities": state.get("resolved_entities", {})}

        resolved_entities = state.get("resolved_entities", {})
        logger.info(f"Resolved entities for canonicalization: {resolved_entities}")
        canonical_query = self.canonicalize_query(nl_query_msg.content, resolved_entities)

        try:
            sql = await generate_sql_from_nl(canonical_query, schema, messages)
            return {"messages": [AIMessage(content=sql, name="sql_query")],
                    "resolved_entities": resolved_entities}
        except Exception as e:
            logger.exception("Error generating SQL")
            return {"messages": [AIMessage(content=f"Error generating SQL: {e}", name="error")],
                    "resolved_entities": resolved_entities}

    async def check_query_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Checking SQL query")
        log_message_contents(state.get("messages", []), logger, prefix="check_query_node-")
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
        log_message_contents(state.get("messages", []), logger, prefix="execute_query_node-")
        messages = state["messages"]
        sql_msg = find_last_message_by_name(messages, "sql_query")
        if not sql_msg:
            return {"messages": [AIMessage(content="No SQL query found in history to execute.", name="error")]}
        try:
            results = await execute_query(sql_msg.content)
            results_str = json.dumps(results)
            return {"messages": [AIMessage(content=results_str, name="results")]} 
        except Exception as e:
            logger.exception("Error executing query")
            return {"messages": [AIMessage(content=f"Error executing query: {e}", name="error")]} 

    async def fix_query_node(self, state: AgentState) -> Dict[str, Any]:
        logger.warning("Node: Attempting to fix SQL query")
        log_message_contents(state.get("messages", []), logger, prefix="fix_query_node-")
        messages = state["messages"]
        # Prepare context for LLM (if needed for future LLM-based query fixing)
        context_messages = await _prepare_llm_context(messages)
        logger.info(f"Using prepared LLM context with {len(context_messages)} messages for query fixing.")
        log_message_contents(context_messages, logger, prefix="fix_query_node-PreparedContext-")
        invalid_sql_msg = find_last_message_by_name(context_messages, "sql_query")
        error_msg = find_last_message_by_name(context_messages, "error")
        schema = state.get("schema")
        original_nl_query_msg = find_last_human_message(context_messages)
        if not all([invalid_sql_msg, error_msg, schema, original_nl_query_msg]):
            return {"messages": [AIMessage(content="Cannot fix query: Missing context.", name="error")]} 
        try:
            fixed_sql = await attempt_fix_sql(
                invalid_sql=invalid_sql_msg.content,
                error_message=error_msg.content,
                schema=schema,
                original_nl_query=original_nl_query_msg.content
            )
            return {"messages": [AIMessage(content=fixed_sql, name="sql_query")]} 
        except Exception as e:
            logger.exception("Error attempting to fix SQL")
            return {"messages": [AIMessage(content=f"Failed to attempt fix: {e}", name="error")]} 

    async def format_answer_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Formatting final answer")
        log_message_contents(state.get("messages", []), logger, prefix="format_answer_node-")
        messages = state["messages"]
        # Prepare context for LLM
        context_messages = await _prepare_llm_context(messages)
        logger.info(f"Using prepared LLM context with {len(context_messages)} messages for answer formatting.")
        log_message_contents(context_messages, logger, prefix="format_answer_node-PreparedContext-")
        llm = get_llm()
        question_msg = find_last_human_message(context_messages)
        results_msg = find_last_message_by_name(context_messages, "results")
        error_msg = find_last_message_by_name(context_messages, "error")
        if not llm:
            return {"messages": [AIMessage(content="Error: LLM not available to format the final answer.")]} 
        if not question_msg:
            return {"messages": [AIMessage(content="Error: Could not find original question to formulate answer.")]} 
        final_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Jij bent Fred, een hartstochtelijke fan van de voetbalclub Feyenoord uit Rotterdam-Zuid. Jouw doel is om vragen over Feyenoord te beantwoorden. Neem de vraag en de resultaten uit de database om een kort en bondig antwoord op de vraag te formuleren. Als de query geen resultaten opleverde, wees daar dan eerlijk over en zeg dat in je antwoord. Ga nooit antwoorden verzinnen."),
            ("human", "Vraag: {question}\nResultaten: {results}")
        ])
        if error_msg and (not results_msg or context_messages.index(error_msg) > context_messages.index(results_msg)):
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
        log_message_contents(state.get("messages", []), logger, prefix="should_fix_or_execute-")
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
        log_message_contents(state.get("messages", []), logger, prefix="after_execution-")
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
        graph.add_node("clarify", self.clarify_node)
        graph.add_node("generate_query", self.generate_query_node)
        graph.add_node("check_query", self.check_query_node)
        graph.add_node("execute_query", self.execute_query_node)
        graph.add_node("fix_query", self.fix_query_node)
        graph.add_node("format_answer", self.format_answer_node)

        # Define edges
        graph.add_edge(START, "get_schema")
        # After getting schema, clarify entities before generating query
        graph.add_edge("get_schema", "clarify")
        graph.add_conditional_edges(
            "clarify",
            lambda state: "generate_query" if find_last_message_by_name(state["messages"], "clarified") else "clarify",
            {
                "generate_query": "generate_query",
                "clarify": "clarify"  # Stay in clarify until resolved
            }
        )
        # After clarifying, generate the SQL query
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
