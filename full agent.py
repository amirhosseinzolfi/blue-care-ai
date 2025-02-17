from typing import List, Dict, Union, Annotated, TypedDict
import logging
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import MessageGraph, StateGraph, END
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.embeddings import OllamaEmbeddings  # Import OllamaEmbeddings
import json
import os

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_NAME = "gpt-4o-mini"  # Updated model name
BASE_URL = "http://localhost:15203/v1"  # G4F API server URL
MAX_CONTEXT_TOKENS = 4096  # Adjust based on your model's context window

# LangChain Setup
llm = ChatOpenAI(
    base_url=BASE_URL,
    model_name=MODEL_NAME,
    api_key="324",
    temperature=0.5
)

# --- Helper Functions ---
def count_tokens(messages: List[BaseMessage], model_name: str = MODEL_NAME) -> int:
    """
    Estimates the number of tokens in a list of messages.
    """
    try:
        model = ChatOpenAI(
    base_url=BASE_URL,
    model_name=MODEL_NAME,
    api_key="324",
    temperature=0.5
)  # Initialize model for token counting
        return model.get_num_tokens(messages)
    except Exception as e:
        logger.warning(f"Error counting tokens, using rough estimate: {e}")
        # Rough estimate if token counting fails (fallback)
        text_content = "".join(msg.content for msg in messages if msg.content)
        return len(text_content.split())  # Simple word count as a fallback

# --- Short-term Memory Implementation ---

# 1. Editing Message Lists (Reducer Function)
def manage_list(existing: list, updates: Union[list, dict]) -> list:
    """
    Reducer function to manage a list in LangGraph state.
    Demonstrates adding to list and keeping a slice of the list.
    """
    logger.info(f"Managing list with updates: {updates}")
    if isinstance(updates, list):
        # Normal case: add new messages to history
        logger.info("Adding new messages to the list.")
        return existing + updates
    elif isinstance(updates, dict) and updates.get("type") == "keep":
        # Keep a slice of the list
        start = updates.get("from", 0)
        end = updates.get("to")
        logger.info(f"Keeping slice of the list from {start} to {end}.")
        return existing[start:end]
    elif isinstance(updates, dict) and updates.get("type") == "remove_ids":
        ids_to_remove = set(updates.get("ids", []))
        logger.info(f"Removing messages with IDs: {ids_to_remove}")
        return [msg for msg in existing if msg.id not in ids_to_remove]
    else:
        logger.warning(f"Unknown list update type: {updates}. Returning existing list.")
        return existing

# State definition for short-term memory example
class ShortTermMemoryState(TypedDict):
    messages: List[BaseMessage]
    user_id: str

# Node for adding AI message (example node 1)
def node_add_ai_message(state: ShortTermMemoryState):
    logger.info("Executing node_add_ai_message")
    ai_message = AIMessage(content="Hello from the bot! How can I help you?")
    return {"messages": [ai_message]}

# Node for deleting old messages (example node 2 - keeping last 2)
def node_delete_old_messages(state: ShortTermMemoryState):
    logger.info("Executing node_delete_old_messages")
    messages = state['messages']
    if len(messages) > 2:
        delete_messages = [{"type": "keep", "from": -2, "to": None}]  # Keep last 2 messages
        logger.info("Requesting to keep only the last 2 messages.")
        return {"messages": delete_messages}
    else:
        logger.info("Less than 2 messages, not deleting any.")
        return {}  # No update needed

# --- Summarizing Past Conversations ---
class SummarizationState(TypedDict):  # Define the state class directly
    messages: List[BaseMessage]
    summary: str

def summarize_conversation(state: SummarizationState):
    """
    Node to summarize the conversation history.
    """
    logger.info("Executing summarize_conversation node.")
    summary = state.get("summary", "")
    messages = state["messages"]

    if summary:
        summary_message = ChatPromptTemplate.from_messages([
            ("system", "You are a summarization expert. You will be given a summary of a conversation so far and new messages. Please extend the summary to include the new messages."),
            ("user", "Existing Summary:\n{summary}\n\nNew Messages:\n{new_messages}\n\nExtended Summary:")
        ])
        prompt = summary_message.format_messages(summary=summary, new_messages="\n".join([m.content for m in messages[-3:]]))  # Summarize last 3 messages with context
    else:
        summary_message = ChatPromptTemplate.from_messages([
            ("system", "You are a summarization expert. Please create a concise summary of the following conversation."),
            ("user", "Conversation:\n{conversation}\n\nSummary:")
        ])
        prompt = summary_message.format_messages(conversation="\n".join([m.content for m in messages[-3:]]))  # Summarize initial messages

    model = ChatOpenAI(
    base_url=BASE_URL,
    model_name=MODEL_NAME,
    api_key="324",
    temperature=0.5
)
    response = model.invoke(prompt)
    logger.info(f"Generated summary: {response.content}")

    # Delete all but the last 2 messages after summarization (for example - adjust as needed)
    if len(messages) > 2:
        delete_messages_update = {"type": "keep", "from": -2, "to": None}
        logger.info("Requesting to keep only the last 2 messages after summarization.")
        return {"summary": response.content, "messages": delete_messages_update}
    else:
        return {"summary": response.content}  # Just update summary, keep all messages if less than 2

# --- Token-based Message Trimming ---
def trim_conversation_history(messages: List[BaseMessage], max_tokens: int = MAX_CONTEXT_TOKENS):
    """
    Trims the conversation history to stay within the token limit.
    Keeps the last messages to fit within max_tokens.
    """
    logger.info("Trimming conversation history based on tokens.")
    trimmed_messages = messages.copy()  # Start with a copy to avoid modifying original
    total_tokens = count_tokens(trimmed_messages)

    if total_tokens <= max_tokens:
        logger.info("Token count is within limit, no trimming needed.")
        return trimmed_messages

    logger.info(f"Token count exceeds limit ({total_tokens} > {max_tokens}). Trimming messages.")
    while total_tokens > max_tokens and len(trimmed_messages) > 1:  # Keep at least one message
        trimmed_messages = trimmed_messages[1:]  # Remove the oldest message
        total_tokens = count_tokens(trimmed_messages)
        logger.info(f"Removed oldest message, new token count: {total_tokens}")

    logger.info(f"Conversation history trimmed to {len(trimmed_messages)} messages, {total_tokens} tokens.")
    return trimmed_messages

def node_trim_messages(state: ShortTermMemoryState):
    """
    Node to trim messages based on token count.
    """
    logger.info("Executing node_trim_messages.")
    current_messages = state["messages"]
    trimmed_messages = trim_conversation_history(current_messages)
    if trimmed_messages != current_messages:
        logger.info("Conversation history was trimmed.")
        return {"messages": trimmed_messages}
    else:
        logger.info("Conversation history was not trimmed (already within limits).")
        return {}  # No update needed if no trimming occurred

# --- Long-term Memory Implementation ---

# 1. InMemoryStore setup (for demonstration - replace with DB-backed store in production)
embed_model = OllamaEmbeddings(model="nomic-embed-text")  # Using OllamaEmbeddings for embedding

def simple_embed_function(texts: List[str]) -> List[List[float]]:
    """
    Simple embedding function placeholder using OllamaEmbeddings for demonstration.
    Replace with a proper embedding model for production.
    """
    logger.info("Generating embeddings using simple_embed_function (OllamaEmbeddings).")
    embeddings = [embed_model.embed_query(text) for text in texts]  # Ensure each text gets its own embedding
    logger.info("Embeddings generated.")
    return embeddings

store = InMemoryStore(index={"embed": simple_embed_function, "dims": 2})  # Using simple_embed_function

# State definition for long-term memory example
class LongTermMemoryState(TypedDict):
    messages: List[BaseMessage]
    user_id: str
    long_term_memory: str  # Example of storing retrieved long-term memory in state

# Node to store memory
def node_store_memory(state: LongTermMemoryState):
    """
    Node to store information into long-term memory.
    For demonstration, stores the last user message as a memory.
    """
    logger.info("Executing node_store_memory.")
    user_id = state["user_id"]
    namespace = (user_id, "user_profile")  # Namespace based on user ID and memory type
    last_user_message = None
    for msg in reversed(state["messages"]):  # Get the latest user message
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break

    if last_user_message:
        memory_key = f"user_message_{len(state['messages'])}"  # Unique key for each memory
        memory_content = {"user_message": last_user_message, "timestamp": "now"}  # Example memory content
        store.put(namespace, memory_key, memory_content)
        logger.info(f"Stored memory in namespace '{namespace}' with key '{memory_key}': {memory_content}")
        
        # Save memory to a local JSON file
        memory_file = "memory.json"
        if os.path.exists(memory_file):
            with open(memory_file, "r") as f:
                data = json.load(f)
        else:
            data = {}
        ns_key = str(namespace)
        if ns_key not in data:
            data[ns_key] = {}
        data[ns_key][memory_key] = memory_content
        with open(memory_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Memory saved to {memory_file}.")
    else:
        logger.info("No user message found to store in memory.")
    return {}  # No state update needed in this example, memory is stored externally

# Node to retrieve memory
def node_retrieve_memory(state: LongTermMemoryState):
    """
    Node to retrieve information from long-term memory.
    For demonstration, retrieves memories related to "user preferences".
    """
    logger.info("Executing node_retrieve_memory.")
    user_id = state["user_id"]
    namespace = (user_id, "user_profile")
    query = "user preferences and needs"  # Example query for memory retrieval

    retrieved_items = store.search(namespace, query=query)  # Search within namespace
    if retrieved_items:
        retrieved_memory_content = "\n".join([str(item.value) for item in retrieved_items])  # Concatenate retrieved memories
        logger.info(f"Retrieved memories: {retrieved_memory_content}")
        return {"long_term_memory": retrieved_memory_content}  # Store in state for use in later nodes
    else:
        logger.info("No relevant memories found.")
        return {"long_term_memory": "No past preferences found."}  # Default message if no memory

# Node that uses long-term memory to generate a response
def node_generate_response_with_memory(state: LongTermMemoryState):
    """
    Node to generate a response using retrieved long-term memory.
    """
    logger.info("Executing node_generate_response_with_memory.")
    memory_context = state.get("long_term_memory", "No specific memory context.")
    user_message_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message_content = msg.content
            break

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the following long-term memory context to personalize your response:\n{memory_context}"),
        ("user", "{user_message}")
    ])
    prompt = prompt_template.format_messages(memory_context=memory_context, user_message=user_message_content)
    model = ChatOpenAI(
    base_url=BASE_URL,
    model_name=MODEL_NAME,
    api_key="324",
    temperature=0.5
)
    response = model.invoke(prompt)
    logger.info(f"Generated response using memory: {response.content}")
    return {"messages": [AIMessage(content=response.content)]}  # Add AI response to messages

# --- LangGraph Graph Definitions and Execution ---

# 1. Short-term Memory Graph Example
def create_short_term_memory_graph():
    """
    Creates a LangGraph with short-term memory management.
    Demonstrates adding AI message, deleting old messages, and trimming messages by tokens.
    """
    logger.info("Creating Short-term Memory LangGraph.")
    builder = StateGraph(ShortTermMemoryState)

    builder.add_node("add_ai_message", node_add_ai_message)
    builder.add_node("delete_old_messages", node_delete_old_messages)
    builder.add_node("trim_messages", node_trim_messages)

    builder.set_entry_point("add_ai_message")
    builder.add_edge("add_ai_message", "delete_old_messages")
    builder.add_edge("delete_old_messages", "trim_messages")
    builder.add_edge("trim_messages", END)

    graph = builder.compile()
    logger.info("Short-term Memory LangGraph compiled.")
    return graph

# 2. Summarization Graph Example
def create_summarization_graph():
    """
    Creates a LangGraph with conversation summarization.
    """
    logger.info("Creating Summarization LangGraph.")
    builder = StateGraph(SummarizationState)

    builder.add_node("summarize_conversation", summarize_conversation)
    builder.add_node("generate_response", node_add_ai_message)  # Reusing node_add_ai_message for simplicity

    builder.set_entry_point("summarize_conversation")
    builder.add_edge("summarize_conversation", "generate_response")
    builder.add_edge("generate_response", END)

    graph = builder.compile()
    logger.info("Summarization LangGraph compiled.")
    return graph

# 3. Long-term Memory Graph Example
def create_long_term_memory_graph():
    """
    Creates a LangGraph with long-term memory interaction.
    Demonstrates storing, retrieving, and using long-term memory.
    """
    logger.info("Creating Long-term Memory LangGraph.")
    builder = StateGraph(LongTermMemoryState)

    builder.add_node("store_memory", node_store_memory)
    builder.add_node("retrieve_memory", node_retrieve_memory)
    builder.add_node("generate_response_with_memory", node_generate_response_with_memory)

    builder.set_entry_point("store_memory")
    builder.add_edge("store_memory", "retrieve_memory")
    builder.add_edge("retrieve_memory", "generate_response_with_memory")
    builder.add_edge("generate_response_with_memory", END)

    graph = builder.compile()
    logger.info("Long-term Memory LangGraph compiled.")
    return graph

def terminal_chat():
    conversation = []
    long_term_graph = create_long_term_memory_graph()
    user_id = "user123"  # Example user ID
    print("Terminal Chat Interface. Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ("exit", "quit"):
            break
        conversation.append(HumanMessage(content=user_input))
        
        # Build conversation context as simple concatenation of messages
        conversation_str = "\n".join([msg.content for msg in conversation])
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{conversation}")
        ])
        prompt = prompt_template.format_messages(conversation=conversation_str)
        response = llm.invoke(prompt)
        print("AI:", response.content)
        conversation.append(AIMessage(content=response.content))
        
        # Update state and invoke long-term memory graph
        state_long_term = {"messages": conversation, "user_id": user_id}
        result_long_term = long_term_graph.invoke(state_long_term)
        logger.info(f"Long-term Memory Graph Result: {result_long_term}")

# --- Example Execution ---
if __name__ == "__main__":
    # --- Short-term Memory Example ---
    logger.info("\n--- Executing Short-term Memory Graph ---")
    short_term_graph = create_short_term_memory_graph()
    initial_state_short_term = {"messages": [HumanMessage(content="Hello bot!")], "user_id": "user123"}
    result_short_term = short_term_graph.invoke(initial_state_short_term)
    logger.info(f"Short-term Memory Graph Result: {result_short_term}")

    # --- Summarization Example ---
    logger.info("\n--- Executing Summarization Graph ---")
    summarization_graph = create_summarization_graph()
    initial_state_summarization = {"messages": [HumanMessage(content="First message."), AIMessage(content="Bot response 1."), HumanMessage(content="Second message.")], "user_id": "user123"}
    result_summarization = summarization_graph.invoke(initial_state_summarization)
    logger.info(f"Summarization Graph Result: {result_summarization}")

    # --- Long-term Memory Example ---
    logger.info("\n--- Executing Long-term Memory Graph ---")
    long_term_graph = create_long_term_memory_graph()
    initial_state_long_term = {"messages": [HumanMessage(content="I like python and short answers.")], "user_id": "user123"}
    result_long_term = long_term_graph.invoke(initial_state_long_term)
    logger.info(f"Long-term Memory Graph Result: {result_long_term}")

    # --- Example of multiple turns with short term memory ---
    logger.info("\n--- Executing Multiple Turns with Short-term Memory Graph ---")
    short_term_graph_multi_turn = create_short_term_memory_graph()
    state_multi_turn = {"messages": [HumanMessage(content="Hello again!")], "user_id": "user123"}

    for i in range(3):  # Run a few turns to see memory in action
        logger.info(f"\n--- Turn {i+1} ---")
        result_turn = short_term_graph_multi_turn.invoke(state_multi_turn)
        logger.info(f"Turn {i+1} Result: {result_turn}")
        state_multi_turn = result_turn  # Update state for next turn (important for memory persistence)
        state_multi_turn["messages"].append(HumanMessage(content=f"User message in turn {i+2}."))  # Simulate next user message

    logger.info("\n--- End of Memory Examples ---")

    # --- Terminal Chat Example ---
    logger.info("\n--- Starting Terminal Chat ---")
    terminal_chat()