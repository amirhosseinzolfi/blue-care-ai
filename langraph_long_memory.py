import logging
from typing import Annotated, TypedDict, Union, List, Dict, Optional, Any  # add import for Any
import operator

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
    RemoveMessage # Import RemoveMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain.memory import ConversationBufferMemory #For conversation history
from langchain_core.runnables import chain
#add more imports needed
from langgraph.store.memory import InMemoryStore # for storing long term memories

# Define proper reducer functions
def concatenate_lists(a: list, b: list) -> list:
    return a + b

def add_strings(a: str, b: str) -> str:
    if a is None:
        return b
    if b is None:
        return a
    return a + b

def merge_dicts(a: Dict, b: Dict) -> Dict:
    if a is None:
        return b
    if b is None:
        return a
    return {**a, **b}

AgentAction = Any  # define AgentAction as Any

# --- (Hypothetical) Existing tools setup ---
def search(query: str) -> str:
    """Searches the web."""
    return f"Search results for '{query}'..."

def summarize_text(text: str) -> str:
    """Summarizes the given text."""
    return f"Summary of '{text[:50]}...'"

tools = [search, summarize_text]
tool_executor = ToolExecutor(tools)

# --- (Hypothetical) Existing LLM setup ---
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125") # Or your preferred model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class State(TypedDict):
    """
    Represents the state of our conversation graph.
    """
    messages: Annotated[List[BaseMessage], concatenate_lists]
    # adding last k messages for short term memory implementation
    last_k_messages: List[BaseMessage]  # No reducer needed for this
    # The user's input
    input: str
    # The agent's previous steps in case of multi-agent and multi-hop graph usecase
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], concatenate_lists] = None
    # The 'intermediate steps' of our agent, will be used for long term memory
    summary: Annotated[str, add_strings] = None # Summary of the conversation so far.
    # Long-term memory namespace
    long_term_memory_namespace: tuple = None
    # Storing conversation context, for short term memory implementation
    conversation_context:str = None

    # We will use this annotated field to manage the long term memory
    long_term_memory: Annotated[Dict, merge_dicts] = None

# --- Helper Functions ---

def trim_messages_history(state: State, k: int = 5) -> Dict:
    """
    Trims the message history to keep only the last 'k' messages, and add to short term memory .

    Args:
        state: The current state.
        k: The number of most recent messages to keep.

    Returns:
        A dictionary containing the trimmed messages.
    """
    logging.info(f"Trimming message history to the last {k} messages.")
    messages = state['messages']
    trimmed_messages = messages[-k:]  # Keep the last k messages

    logging.info(f"Trimmed messages: {trimmed_messages}")
        # Add the trimmed messages to the state under the "last_k_messages" key.
    return {"last_k_messages": trimmed_messages}

# Summarization is triggered based on token count or explicit calls
def summarize_memory(state: State, llm: ChatOpenAI, max_token_limit: int = 1000) -> dict:
    """Summarizes the conversation history if it's too long.

    Args:
        state: The current state.
        llm: The language model to use for summarization.
        max_token_limit:  A rough estimate of the max tokens before summarizing.

    Returns:
        A dictionary with the updated 'summary' and potentially removed messages.
    """
    messages = state['messages']
    current_summary = state.get("summary", "")

    # Very basic token count estimation (replace with a proper tokenizer if needed)
    estimated_tokens = sum(len(msg.content.split()) for msg in messages)

    if estimated_tokens > max_token_limit:
        logging.info("Summarizing conversation history due to token limit.")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant tasked with summarizing conversations. "
                       "Maintain as much relevant information as possible, "
                       "prioritizing the most recent interactions."),
            ("user", "Here's the conversation history:\n\n{history}"),
            ("user", "Please provide a concise summary."),
        ])

        # Format the prompt with the current message history
        formatted_prompt = prompt.format_prompt(history="\n".join([f"{msg.type}: {msg.content}" for msg in messages]))

        # Invoke the LLM to generate the summary
        summary_response = llm.invoke(formatted_prompt)
        new_summary = summary_response.content

        logging.info(f"New summary generated: {new_summary}")

        # Create RemoveMessage objects for all but the last 2 messages.
        messages_to_remove = [RemoveMessage(id=m.id) for m in messages[:-2] if hasattr(m, 'id') and m.id]

        return {
            "summary": new_summary,
            "messages": messages_to_remove  # Remove old messages, keep the last two
        }
    else:
         logging.info(f"Estimated token count ({estimated_tokens}) below limit. No summarization needed.")
         return {"summary": current_summary}


def prepare_messages_for_llm(state: State) -> List[BaseMessage]:
    """
    Prepares the messages to be sent to the LLM.  Combines:
        - System Message (if any)
        - Last 'k' messages (from trim_messages)
        - Summary (if any)
    """
    logging.info("Preparing messages for LLM input.")

    messages_for_llm = []

    # Add system message if you have one defined at the start of your conversation.
    # messages_for_llm.append(SystemMessage(content="You are a helpful assistant."))

    # Add the summary if it exists
    if state.get("summary"):
      messages_for_llm.append(SystemMessage(content=f"Conversation Summary:\n{state['summary']}"))

    # Retrieve the last 'k' messages, we are retrieving shot term memory here
    last_k_messages = state.get("last_k_messages", [])
    messages_for_llm.extend(last_k_messages)
    # Add the latest user input.
    messages_for_llm.append(HumanMessage(content=state["input"]))

    logging.info(f"Prepared messages for LLM: {messages_for_llm}")
    return messages_for_llm

def initialize_long_term_memory(state: State) -> Dict:
    """
    Initializes the long-term memory store and sets the namespace.  This is
    typically done *once* at the start of a user's interaction.

    Args:
        state: The current state.

    Returns:
        A dictionary containing the initialized long-term memory namespace.
    """
    user_id = "user_123"  # Replace with actual user ID from your application
    application_context = "customer_support"  # Or whatever is relevant
    namespace = (user_id, application_context)
    logging.info(f"Initializing long-term memory for namespace: {namespace}")

    # Initialize the long-term memory store (if not already done)
    if state.get("long_term_memory") is None:
       store = InMemoryStore()
       return {"long_term_memory_namespace": namespace , "long_term_memory" : store }
    else:
      return {"long_term_memory_namespace": namespace}

def save_long_term_memory(state: State, memory_key: str, data: Dict) -> Dict:
    """
    Saves data to long-term memory.

    Args:
        state: The current state.
        memory_key: A unique key within the namespace for this memory.
        data: The data to store (must be JSON serializable).

    Returns:
        An empty dictionary (state updates are handled by the store).
    """
    namespace = state["long_term_memory_namespace"]
    store = state["long_term_memory"]
    logging.info(f"Saving to long-term memory under namespace {namespace}, key {memory_key}.")
    store.put(namespace, memory_key, data)

    return {}  # State update is implicit via the store

def retrieve_long_term_memory(state: State, memory_key: str) -> Dict:
    """
    Retrieves data from long-term memory.

    Args:
        state: The current state.
        memory_key: The key of the memory to retrieve.

    Returns:
        A dictionary containing the retrieved data, or an empty dictionary if not found.
    """
    namespace = state["long_term_memory_namespace"]
    store = state["long_term_memory"]
    logging.info(f"Retrieving from long-term memory under namespace {namespace}, key {memory_key}.")
    try:
        retrieved_data = store.get(namespace, memory_key)
        logging.info(f"Retrieved data: {retrieved_data}")
        # we can return and store data in state if we want to use them
        return {"retrieved_long_term_data": retrieved_data}
    except KeyError:
        logging.warning(f"Memory key '{memory_key}' not found in namespace {namespace}.")
        return {}


def update_instructions(state: State) -> Dict:
    """
    Example of updating instructions based on conversation history, a form of
    procedural long-term memory.  This is a simplified example.
    """
    store = state["long_term_memory"]
    namespace = ("instructions",)  # Using a separate namespace for instructions
    logging.info("Updating agent instructions based on conversation.")

    #--- define a prompt ---
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Here are the current instructions:\n{instructions}\n\n"
                "Based on the following conversation, update the instructions:\n{conversation}",
            ),
            ("user", "Suggest new instructions:"),
        ]
    )

    # 1. Retrieve current instructions
    try:
        current_instructions = store.get(namespace, "agent_a")["instructions"]
    except (KeyError, TypeError):
        current_instructions = "Be helpful and concise."  # Default instructions

    # 2. Format the prompt
    formatted_prompt = prompt_template.format_prompt(
        instructions=current_instructions, conversation=state["messages"]
    )

    # 3. Invoke the LLM
    response = model.invoke(formatted_prompt)
    new_instructions = response.content

    # 4. Store the updated instructions
    logging.info(f"New instructions: {new_instructions}")
    store.put(namespace, "agent_a", {"instructions": new_instructions})

    return {}

def agent(state: State) -> Dict:
    """
    The main agent function, responsible for processing input and deciding actions.
    """

    # Prepare messages for LLM (including short-term memory)
    messages = prepare_messages_for_llm(state)

    # Invoke the LLM
    response = model.invoke(messages)
    logging.info(f"Agent received LLM response: {response}")

    # Check and update the intermediate steps, used for long term memory
    # Check if 'intermediate_steps' exists and is not None, otherwise initialize it
    if state.get('intermediate_steps') is None:
        intermediate_steps = []
    else:
        intermediate_steps = state['intermediate_steps']

    return {"messages": [response],"intermediate_steps": intermediate_steps}
def tool_node(state: State) -> Dict:
    """
    Executes a tool based on the agent's output.
    """
    logging.info("Executing tool node.")
    messages = state['messages']
    last_message = messages[-1]

    # Check and update the intermediate steps, used for long term memory
    # Check if 'intermediate_steps' exists and is not None, otherwise initialize it
    if state.get('intermediate_steps') is None:
        intermediate_steps = []
    else:
        intermediate_steps = state['intermediate_steps']

    # We assume the last message is the tool call
    if isinstance(last_message, FunctionMessage):
        raise ValueError("The last message should be a tool call, not a tool response.")

    tool_name = last_message.additional_kwargs["tool_calls"][0]["function"]["name"]
    tool_input = last_message.additional_kwargs["tool_calls"][0]["function"]["arguments"]
     # Construct a ToolInvocation from the model's response
    tool_action = ToolInvocation(
            tool=tool_name,
            tool_input=tool_input,
        )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(tool_action)

    # We return a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=tool_name)

    return {"messages": [function_message],"intermediate_steps": intermediate_steps}
    #return {"messages": [function_message]}

def should_continue(state: State) -> str:
    """
    Decides whether to continue the conversation or end it.
    """
    messages = state['messages']
    last_message = messages[-1]
    # If there is no function call, then we finish the conversation
    if "tool_calls" not in last_message.additional_kwargs:
        logging.info("Ending conversation (no tool call).")
        return "end"
    # Otherwise if there is, we continue the conversation
    else:
        logging.info("Continuing conversation (tool call detected).")
        return "continue"

# --- Building the Graph ---
workflow = StateGraph(State)

# --- add nodes to graph ---
workflow.add_node("initialize_memory", initialize_long_term_memory)  # Initialize memory
workflow.add_node("agent", agent)  # The main agent
workflow.add_node("tool", tool_node)  # Tool execution
workflow.add_node("trim_messages", trim_messages_history)  # Trim messages
workflow.add_node("summarize", summarize_memory)  # Summarize conversation
workflow.add_node("update_instructions", update_instructions) # update instructions


# --- set graph edges ---

# Start with memory initialization, and always initialize long-term memory
workflow.set_entry_point("initialize_memory")
workflow.add_edge("initialize_memory", "trim_messages")

# Then, trim messages before each agent run
workflow.add_edge("trim_messages", "summarize")

# Summarize the conversation before each agent run, if needed.
workflow.add_edge("summarize", "agent")

# Connect agent to tool node, and tool node back to itself
workflow.add_conditional_edges("agent", should_continue, {
    "continue": "tool",
    "end": "update_instructions" # Update instructions at the end,
})
workflow.add_edge("tool", "trim_messages")


# update instructions at the end of graph
workflow.add_edge("update_instructions", END)

# Compile the graph
app = workflow.compile()


# Example Usage

inputs = {
    "input": "What's the weather like in London?",
     "messages": [] # Start with an empty message list
}

for output in app.stream(inputs):
    for key, value in output.items():
        if key != "__end__":  # Don't print the final output
            print(f"Output from node '{key}':")
            print("---")
            print(value)
            print("---")
    print("\n---\n")

# --- Example of saving and retrieving specific data ---

# After the conversation, save some specific information to long-term memory
app.invoke({
    "input": "Save user preference: prefers metric units.",
    "messages": [HumanMessage(content="User prefers metric units.")],
    "intermediate_steps": [], # Provide an empty list or appropriate value
    "long_term_memory_namespace": ("user_123", "customer_support"),
     "long_term_memory" : InMemoryStore() # Initialize the long-term memory store
}) # initialize long-term-memory, you can run graph again
save_long_term_memory(app.get_state(), "user_preferences", {"unit_system": "metric"})

# Later, retrieve the information
retrieved_state = retrieve_long_term_memory(app.get_state(), "user_preferences")
if retrieved_state["retrieved_long_term_data"]:
    print("Retrieved user preferences:", retrieved_state["retrieved_long_term_data"])

# you can add store.search for implementing RAG and retreival of informations