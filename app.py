import chainlit as cl
import logging
from langchain_core.messages import HumanMessage, AIMessage
from nl2sql.src.workflow.manager import WorkflowManager
from nl2sql.src.workflow import config

# --- Configuration and Logging ---
# The config module now handles loading the .env file.
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Workflow Initialization ---
# Create a configuration dictionary from the imported config module
try:
    # Instantiate the manager with the app config.
    workflow_manager = WorkflowManager(format_output=True)
    workflow = workflow_manager.get_graph()
    logger.info("WorkflowManager initialized and graph compiled successfully.")
except Exception as e:
    logger.exception("Fatal error during WorkflowManager initialization.")
    workflow = None

# Authentication callback disabled for now
#@cl.oauth_callback
#def oauth_callback(
#  provider_id: str,
#  token: str,
#  raw_user_data: Dict[str, str],
#  default_user: cl.User,
#) -> Optional[cl.User]:
#  return default_user

@cl.on_chat_start
async def on_chat_start():
    if not workflow:
        await cl.Message(content="Sorry, the chatbot is not available due to a configuration error. Please contact the administrator.").send()
        return
    
    cl.user_session.set("workflow", workflow)
    # Initialize message history
    cl.user_session.set("messages", []) 
    logger.info("Chat session started and workflow stored.")

@cl.on_message
async def on_message(message: cl.Message):
    logger.info(f"Received message: '{message.content}'")
    
    # Retrieve the compiled graph from the session
    compiled_workflow = cl.user_session.get("workflow")
    if not compiled_workflow:
        logger.error("Workflow not found in user session.")
        await cl.Message(content="Sorry, there seems to be a session error. Please restart the chat.").send()
        return

    # Set up configuration for the LangGraph invocation
    config = {"configurable": {"thread_id": cl.context.session.id}}
    
    # Retrieve message history and add the new message
    messages = cl.user_session.get("messages", [])
    messages.append(HumanMessage(content=message.content))

    # Define the initial state for the workflow run
    initial_state = { "messages": messages }
    logger.debug(f"Invoking workflow with initial state: {initial_state}")

    final_state = None
    try:
        # Stream the output of the workflow
        async for output in compiled_workflow.astream(initial_state, config=config):
            # The final state is the last item in the stream
            final_state = output
            
        logger.info("Workflow stream finished.")
        logger.debug(f"Final state: {final_state}")

    except Exception as e:
        logger.exception("An error occurred during workflow invocation.")
        await cl.Message(content=f"An error occurred: {e}").send()
        return

    # Extract the final state dictionary from the stream output
    if not final_state or not isinstance(final_state, dict):
         logger.error(f"Workflow did not return a valid final state. Got: {final_state}")
         await cl.Message(content="An unknown error occurred.").send()
         return

    final_state_data = list(final_state.values())[0]

    # Persist the full message history for the next turn
    cl.user_session.set("messages", final_state_data.get("messages", []))

    # Display the natural language answer from the last AI message
    last_message = final_state_data.get("messages", [])[-1]
    if isinstance(last_message, AIMessage):
         answer = last_message.content
         logger.info(f"Sending final answer: {answer}")
         await cl.Message(content=answer).send()
    else:
        logger.error(f"Final message in state was not an AIMessage: {last_message}")
        await cl.Message(content="Sorry, I couldn't formulate a proper response.").send()
