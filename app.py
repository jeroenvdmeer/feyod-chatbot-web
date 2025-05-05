import chainlit as cl
from agent.WorkflowManager import WorkflowManager
import logging
from common import config
from langchain_core.messages import HumanMessage, AIMessage

# --- Logging Setup ---
# Configure logging using the level from config
log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

@cl.on_chat_start
def on_chat_start():
    logger.info("Chat started. Initializing workflow.")
    try:
        workflow = WorkflowManager().compile_graph()
        cl.user_session.set("workflow", workflow)
        logger.info("Workflow compiled and stored in user session.")
    except Exception as e:
        logger.exception("Failed to initialize workflow on chat start.")
        # Optionally send a message to the user, though chat hasn't fully started

@cl.on_message
async def on_message(message: cl.Message):
    logger.info(f"Received message: {message.content}")
    workflow = cl.user_session.get("workflow")
    if not workflow:
        logger.error("Workflow not found in user session.")
        await cl.Message(content="Sorry, de workflow is niet correct geïnitialiseerd. Probeer de chat opnieuw te starten.").send()
        return

    # Initialize state ONLY with the messages list containing the HumanMessage
    initial_state = {
        "messages": [HumanMessage(content=message.content)]
    }
    logger.debug(f"Initial state for workflow: {initial_state}")

    final_state = None
    logger.info("Invoking workflow...")
    try:
        # Use ainvoke to get the final state directly
        final_state = await workflow.ainvoke(initial_state)
        logger.info("Workflow invocation finished.")
        logger.debug(f"Final state received: {final_state}") # Log final state at DEBUG

    except Exception as e:
        logger.exception("An error occurred during workflow invocation.")
        await cl.Message(content=f"Er is een fout opgetreden tijdens het uitvoeren van de workflow: {e}").send()
        return

    # The final state from ainvoke IS the complete state dictionary
    if not final_state or not final_state.get("messages"):
        logger.error(f"Workflow did not return messages in final state. Final state: {final_state}")
        await cl.Message(content="Er is een onbekende fout opgetreden na het uitvoeren van de workflow.").send()
        return

    # Display the natural language answer from the last message in the final state
    # Ensure the last message is indeed an AIMessage (it should be from format_answer_node)
    if final_state["messages"] and isinstance(final_state["messages"][-1], AIMessage):
         answer = final_state["messages"][-1].content
         logger.info(f"Sending final answer: {answer}")
         await cl.Message(content=answer).send()
    else:
        # Handle cases where the last message might not be the AI's answer (e.g., unexpected error)
        last = final_state['messages'][-1] if final_state['messages'] else 'None'
        logger.error(f"Final message in state is not an AIMessage: {last}")
        await cl.Message(content="Sorry, ik kon het antwoord niet correct formatteren.").send()
