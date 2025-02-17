
import logging
from typing import Annotated, List, Union, Dict, TypedDict
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import StateCheckpoint
from langgraph.pregel import PregelProcess
from langgraph.graph import add_messages
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_community.utilities import TavilySearchAPIWrapper

from langgraph.store.memory import InMemoryStore, BaseStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User provided API keys (sensitive information, handle securely in real applications)
GEMINI_API_KEY = "AIzaSyB3HCPQkJCHiUbxamxLybRW9u6PWOfLmKs"
TAVILY_API_KEY = "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"

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


# 1. Initialize Language Model
logger.info("Step 1: Initializing Language Model (Gemini)")
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)

# 2. Short-term Memory Functions

## 2.1. Editing Message Lists - `manage_list` function (example, not directly used in graph but shown for concept)
logger.info("Step 2.1: Defining `manage_list` function (Short-term Memory - Editing Lists)")
def manage_list(existing: list, updates: Union[list, dict]):
    if isinstance(updates, list):
        return existing + updates
    elif isinstance(updates, dict) and updates["type"] == "keep":
        return existing[updates["from"]:updates["to"]]
    return existing # Default return if not matched

## 2.2. State Definition for Short-term Memory
logger.info("Step 2.2: Defining State for Short-term Memory")
class ShortTermMemoryState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # Using add_messages directly
    # Example of using manage_list - not used directly in graph for simplicity but showing how it can be defined
    # my_list: Annotated[list, manage_list] # Example using manage_list if needed for custom list management


## 2.3. Node for Removing Messages (Short-term Memory - Removing Messages)
logger.info("Step 2.3: Defining Node for Removing Messages")
def remove_old_messages(state: ShortTermMemoryState):
    logger.info("Executing `remove_old_messages` node.")
    messages = state.get("messages", [])
    if len(messages) > 3: # Keep last 3 messages for context, remove older ones
        logger.info("Conversation history is long. Removing older messages.")
        delete_messages = [RemoveMessage(id=m.id) for m in messages[:-3]] # Remove all but last 3
        return {"messages": delete_messages}
    else:
        logger.info("Conversation history is short, keeping all messages.")
        return {} # No messages to remove


## 2.4. Node for Summarizing Conversation (Short-term Memory - Summarization)
logger.info("Step 2.4: Defining Node for Summarizing Conversation")
def summarize_conversation(state: ShortTermMemoryState):
    logger.info("Executing `summarize_conversation` node.")
    messages = state.get("messages", [])
    if not messages:
        logger.info("No messages to summarize.")
        return {}

    # Simple summarization prompt - can be improved
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following conversation to maintain context within token limits. Focus on key topics and user intents."),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Please provide a concise summary of the conversation above.")
    ])
    summarization_chain = prompt_template | model | StrOutputParser()

    logger.info("Summarizing conversation history.")
    summary = summarization_chain.invoke({"messages": messages})
    logger.info(f"Conversation Summary: {summary}")

    # After summarization, let's keep only the summary and the latest user message to reduce context length drastically
    if len(messages) > 2: # Summarize only if conversation is longer than 2 messages (summary + latest user)
        logger.info("Conversation summarized. Removing older messages and keeping summary.")
        latest_user_message = messages[-1] if messages else None # Keep latest user message if available

        summarized_messages = [
            AIMessage(content=f"Conversation Summary: {summary}"),
        ]
        if latest_user_message:
            summarized_messages.append(latest_user_message)

        delete_messages = [RemoveMessage(id=m.id) for m in messages[:-len(summarized_messages)] if m not in summarized_messages]

        return {"messages": summarized_messages + delete_messages} # Replace with summary and delete older messages
    else:
        logger.info("Conversation is short, skipping summarization for now.")
        return {} # No summarization needed yet


## 2.5. Node for Responding to User (Agent Logic - Example using Tavily for search)
logger.info("Step 2.5: Defining Node for Responding to User (Agent Logic)")
def generate_response(state: ShortTermMemoryState):
    logger.info("Executing `generate_response` node.")
    messages = state.get("messages", [])
    latest_message = messages[-1].content if messages else "Hello" # Default greeting if no message

    # Simple agent logic: if user asks a question, use Tavily Search
    if "?" in latest_message:
        logger.info("User asked a question. Using Tavily Search.")
        search = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)
        search_results = search.run(latest_message)
        response_content = f"Search Results: {search_results}"
    else:
        response_content = "Hello! How can I assist you today?"

    response_message = AIMessage(content=response_content)
    logger.info(f"Generated Response: {response_content}")
    return {"messages": [response_message]}


# 3. Long-term Memory Implementation

## 3.1. Initialize Long-term Memory Store
logger.info("Step 3.1: Initializing Long-term Memory Store (InMemoryStore)")
store: BaseStore = InMemoryStore() # Using InMemoryStore for simplicity
user_id = "user123" # Example user ID
application_context = "agent_context_v1" # Example context
namespace = (user_id, application_context)

## 3.2. Function to Store Long-term Memory (Example: User Preferences)
logger.info("Step 3.2: Defining Function to Store Long-term Memory")
def store_user_preference(preference_key: str, preference_value: str):
    logger.info(f"Storing long-term memory - Preference: {preference_key}: {preference_value}")
    memory_key = f"user_preference_{preference_key}"
    store.put(namespace, memory_key, {"preference": preference_value})

## 3.3. Function to Retrieve Long-term Memory (Example: User Preferences)
logger.info("Step 3.3: Defining Function to Retrieve Long-term Memory")
def retrieve_user_preference(preference_key: str) -> Union[str, None]:
    logger.info(f"Retrieving long-term memory - Preference: {preference_key}")
    memory_key = f"user_preference_{preference_key}"
    retrieved_memory = store.get(namespace, memory_key)
    if retrieved_memory and "preference" in retrieved_memory:
        return retrieved_memory["preference"]
    return None

## 3.4. Node to Integrate Long-term Memory (Example: Greeting with User Preference)
logger.info("Step 3.4: Defining Node to Integrate Long-term Memory")
def generate_response_with_long_term_memory(state: ShortTermMemoryState):
    logger.info("Executing `generate_response_with_long_term_memory` node.")
    messages = state.get("messages", [])
    latest_message_content = messages[-1].content if messages else ""

    user_name_preference = retrieve_user_preference("name") # Example: Retrieve user's preferred name

    greeting = "Hello!"
    if user_name_preference:
        greeting = f"Hello {user_name_preference}!"

    # Agent logic similar to generate_response but incorporating long-term memory
    if "?" in latest_message_content:
        logger.info("User asked a question. Using Tavily Search.")
        search = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)
        search_results = search.run(latest_message_content)
        response_content = f"{greeting} Search Results: {search_results}"
    else:
        response_content = f"{greeting} How can I assist you today?"

    response_message = AIMessage(content=response_content)
    logger.info(f"Generated Response with Long-term Memory: {response_content}")
    return {"messages": [response_message]}

## 3.5. Node to Store New Long-term Memory (Example: Storing User Name)
logger.info("Step 3.5: Defining Node to Store New Long-term Memory")
def store_new_memory(state: ShortTermMemoryState):
    logger.info("Executing `store_new_memory` node.")
    messages = state.get("messages", [])
    latest_user_message_content = messages[-1].content if messages else ""

    if "my name is" in latest_user_message_content.lower():
        name = latest_user_message_content.split("my name is")[-1].strip()
        if name:
            logger.info(f"Detected user name: {name}. Storing in long-term memory.")
            store_user_preference("name", name)
            return {"messages": [AIMessage(content=f"Okay, I will remember your name is {name}.")]} # Acknowledge storing name
    return {} # No new memory to store in this case


# 4. LangGraph Agent Definition

## 4.1. Define the StateGraph
logger.info("Step 4.1: Defining the StateGraph")
workflow = StateGraph(ShortTermMemoryState)

## 4.2. Add Nodes to the Graph
logger.info("Step 4.2: Adding Nodes to the Graph")
workflow.add_node("generate_response", generate_response_with_long_term_memory) # Using long-term memory version
workflow.add_node("remove_messages", remove_old_messages)
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_node("store_memory", store_new_memory) # Node to store long-term memory

## 4.3. Define Edges - Flow Control
logger.info("Step 4.3: Defining Edges and Flow Control")
workflow.set_entry_point("generate_response") # Start with response generation

# Conditional edge for summarization - summarize if conversation gets long (example condition)
def should_summarize(state):
    messages = state.get("messages", [])
    if len(messages) > 5: # Example: Summarize if more than 5 messages
        logger.info("Condition met: Conversation is long, proceeding to summarization.")
        return "summarize"
    else:
        logger.info("Condition not met: Conversation is short, skipping summarization.")
        return "no_summarize"

workflow.add_conditional_edges(
    "generate_response",
    should_summarize,
    {
        "summarize": "summarize_conversation",
        "no_summarize": "store_memory" # If no summarize, go to store memory check
    }
)
workflow.add_edge("summarize_conversation", "store_memory") # After summarization, check for new memories
workflow.add_edge("store_memory", "remove_messages") # After storing, remove old messages if needed
workflow.add_edge("remove_messages", "generate_response") # Loop back to generate response after cleanup


# Add an end node (optional for simple loops but good practice for complex graphs)
workflow.add_node("end", END)
workflow.add_edge("remove_messages", END) # Example: End after message removal - adjust as needed based on desired flow

## 4.4. Compile the Graph
logger.info("Step 4.4: Compiling the StateGraph")
app = workflow.compile()

# 5. Run the Agent and Test Memory

## 5.1. Example Conversation Loop
logger.info("Step 5.1: Starting Example Conversation Loop")
user_messages = [
    "Hello",
    "What is the weather in London?",
    "My name is Alice", # Example to trigger long-term memory storage
    "Okay, what time is it?",
    "Tell me a joke",
    "What did you say my name was?", # Test long-term memory retrieval
    "Thank you!",
]

# Initialize state outside the loop
current_state = {"messages": []} # Initial empty message list

for user_message in user_messages:
    logger.info(f"\nUser Input: {user_message}")
    current_state["messages"].append(HumanMessage(content=user_message))

    # Run the graph, passing in the current state
    try:
        result = app.invoke(current_state, config=RunnableConfig(tags=["example_session"]))
        current_state = result # Update state with the output of the graph run
        print("-" * 40) # Separator for clarity
        print(f"Agent Response: {result['messages'][-1].content if result['messages'] else 'No response'}") # Print last agent message
        print("-" * 40)

    except Exception as e:
        logger.error(f"Error during agent execution: {e}")
        break # Exit loop on error for example purposes

logger.info("Example conversation finished.")
print("\n--- End of Conversation ---")
