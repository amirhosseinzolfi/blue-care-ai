import uuid
import json
import logging
import os
import datetime
import ast  # Added for safer literal evaluation
from rich.logging import RichHandler
from rich import print as rprint
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from typing import TypedDict, List, Dict, Optional
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from langchain.output_parsers import OutputFixingParser

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("long-memory-agent")
rprint("[bold blue]Logging is set up.[/bold blue]")  # New pretty log

# Define the state schema
class State(TypedDict):
    user_input: str
    conversation_history: List[Dict[str, str]]  # Store messages as dicts
    extracted_memory: Optional[Dict]  # Store extracted memory

# Initialize the LLM
llm = ChatOpenAI(
    base_url="http://localhost:15203/v1",  # Or your actual OpenAI endpoint
    model_name="gpt-4o",  # Replace with your model
    api_key="your_api_key",  # Replace with your key
    temperature=0.5
)
logger.info("LangChain LLM initialized.")
rprint("[bold green]LangChain LLM is online.[/bold green]")  # New pretty log

# MongoDB setup for long-term memory
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["memory_db"]
memory_collection = db["memories"]
logger.info("MongoDB connected.")
rprint("[bold green]Connected to MongoDB memory_db.[/bold green]")  # New pretty log

# JSON file for local memory backup
LOCAL_MEMORY_FILE = "long_term_memory.json"

def initialize_memory_file():
    """Initialize the memory file with default structure if it doesn't exist."""
    default_structure = {
        "1": [],  # Default user ID
        "metadata": {
            "created_at": datetime.datetime.now().isoformat(),
            "version": "1.0"
        }
    }
    try:
        if not os.path.exists(LOCAL_MEMORY_FILE):
            with open(LOCAL_MEMORY_FILE, 'w') as f:
                json.dump(default_structure, f, indent=4)
            logger.info(f"Created new memory file: {LOCAL_MEMORY_FILE}")
            rprint(f"[bold green]Created new memory file: {LOCAL_MEMORY_FILE}[/bold green]")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize memory file: {e}")
        rprint(f"[bold red]Failed to initialize memory file: {e}[/bold red]")
        return False

def load_local_memory():
    """Load or create local memory file."""
    try:
        # First ensure the file exists
        initialize_memory_file()
        
        with open(LOCAL_MEMORY_FILE, "r") as f:
            data = json.load(f)
            logger.info("Local memory loaded successfully")
            rprint("[bold green]Local memory loaded successfully[/bold green]")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in memory file: {e}")
        rprint("[bold red]Invalid JSON in memory file, creating new one[/bold red]")
        initialize_memory_file()
        return {"1": []}
    except Exception as e:
        logger.error(f"Error loading memory: {e}")
        rprint(f"[bold red]Error loading memory: {e}[/bold red]")
        return {"1": []}

def save_local_memory(memory_data):
    with open(LOCAL_MEMORY_FILE, "w") as f:
        json.dump(memory_data, f, indent=4)  # Use indent for pretty printing
    logger.info("Local memory saved to JSON.")

# Load initial local memory
local_memory = load_local_memory()
rprint("[bold blue]Initial local memory loaded.[/bold blue]")  # New pretty log

# Tool for extracting structured data (example)
def extract_data(text: str) -> Dict:
    """Extracts structured data from text (example implementation)."""
    try:
        lines = text.strip().split('\n')
        data = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()
        return data
    except Exception as ex:
        logger.error(f"Extraction error: {ex}")
        return {}  # Return empty if extraction fails

tools = [
    Tool(
        name="DataExtractor",
        func=extract_data,
        description="Useful for extracting structured data from text.  Input should be the text to extract from.",
    )
]

# Initialize the agent with handle_parsing_errors
memory_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history"),
    prompt=MessagesPlaceholder(variable_name="chat_history"),
    handle_parsing_errors=True,  # Add this parameter
    max_iterations=3  # Add retry limit
)
logger.info("Memory agent initialized.")
rprint("[bold green]Memory agent is ready.[/bold green]")  # New pretty log

# Define the nodes in the graph
def process_message(state: State) -> Dict:
    logger.info("Starting process_message.")
    rprint("[bold blue]Processing new message...[/bold blue]")  # New pretty log
    user_input = state["user_input"]
    conversation_history = state["conversation_history"]
    
    # Ensure local_memory is accessed as a global variable
    global local_memory, user_id

    conversation_history.append({"role": "user", "content": user_input})  # Add to history
    logger.info("User input appended to history.")

    try:
        # Updated to properly handle agent response
        response = memory_agent.invoke(
            {"input": user_input},
            {"handle_parsing_errors": True}
        )
        agent_response = response.get('output', 
            "I apologize, but I couldn't parse that properly. Could you rephrase?")
    except Exception as e:
        logger.error(f"Agent response error: {e}")
        agent_response = "I encountered an error processing that. Could you try again?"

    logger.info("Received response from memory agent.")
    rprint(f"[italic green]Agent response:[/italic green] {agent_response}")  # New pretty log

    conversation_history.append({"role": "assistant", "content": agent_response})  # Add to history

    # Modified memory storage logic
    if "save this" in user_input.lower() or "remember" in user_input.lower():
        try:
            rprint("[bold yellow]Extracting memory...[/bold yellow]")
            
            # Store the raw input as memory if it contains "save this:"
            if "save this:" in user_input.lower():
                memory_content = user_input.split("save this:", 1)[1].strip()
            else:
                memory_content = user_input

            # Create memory entry
            memory_entry = {
                "content": memory_content,
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "user_memory"
            }
            
            # Make sure the user_id exists in local_memory
            if user_id not in local_memory:
                local_memory[user_id] = []
            
            # Append the new memory
            local_memory[user_id].append(memory_entry)
            
            # Save to file
            save_local_memory(local_memory)
            
            logger.info(f"Saved memory to local JSON: {memory_entry}")
            rprint(f"[bold green]Memory saved:[/bold green] {memory_entry}")
            extracted_memory = memory_entry
            
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            rprint(f"[bold red]Error saving memory: {e}[/bold red]")
            extracted_memory = None
    else:
        extracted_memory = None

    result = {
        "user_input": agent_response,  # Agent's response becomes the next input
        "conversation_history": conversation_history,
        "extracted_memory": extracted_memory
    }
    logger.info("process_message returning result.")
    rprint("[bold blue]Finished processing message.[/bold blue]")  # New pretty log
    return result

# Create the StateGraph (no changes)
# ... (rest of the code is the same)

# Start the process
user_id = "1"  # You can use this for user-specific memory if needed
rprint("[bold underline green]Starting Long Memory Agent[/bold underline green]")

# Load previous conversation history from local json
try:
    if not os.path.exists(LOCAL_MEMORY_FILE):
        initialize_memory_file()
    
    with open(LOCAL_MEMORY_FILE, 'r') as f:
        local_memory = json.load(f)
        initial_history = local_memory.get(user_id, [])
        logger.info("Loaded previous conversation history")
        rprint("[bold green]Loaded previous conversation history[/bold green]")
except Exception as e:
    logger.error(f"Error loading history: {e}")
    rprint(f"[bold red]Error loading history: {e}[/bold red]")
    initial_history = []
    # Initialize with empty structure
    local_memory = {"1": []}
    save_local_memory(local_memory)

initial_state = {
    "user_input": "Hello, I want to save some info. Name: John Doe\nAge: 30",
    "conversation_history": [],
    "extracted_memory": None
}
rprint("[bold blue]Initial state prepared.[/bold blue]")  # New pretty log

# Invoke the graph (no changes)
# ...

# Continue the conversation (no changes)
# ...

rprint("[bold underline green]Process Complete[/bold underline green]")

class TerminalChatTester:
    def __init__(self):
        # Initialize a separate model for testing
        self.test_llm = ChatOpenAI(
            base_url="http://localhost:15203/v1",
            model_name="gpt-4",
            api_key="your_api_key",
            temperature=0.7
        )
        self.conversation_memory = ConversationBufferMemory()
        rprint("[bold blue]Terminal Chat Tester initialized[/bold blue]")
        self.memory_stats = {
            "total_memories": 0,
            "successful_saves": 0,
            "failed_saves": 0
        }
        self.current_session_memories = []
        self.user_profile = {
            "facts": [],
            "preferences": [],
            "language": None
        }
        self.supported_languages = ["en", "fa"]

    def display_memory_stats(self):
        """Display current memory statistics"""
        rprint("\n[bold yellow]===== Memory Stats =====[/bold yellow]")
        rprint(f"Total memories stored: {self.memory_stats['total_memories']}")
        rprint(f"Successful saves: {self.memory_stats['successful_saves']}")
        rprint(f"Failed saves: {self.memory_stats['failed_saves']}")
        rprint("[yellow]=====================[/yellow]\n")

    def display_current_memories(self):
        """Display all memories from current session"""
        with open(LOCAL_MEMORY_FILE, 'r') as f:
            memories = json.load(f)
            rprint("\n[bold cyan]===== Current Memories =====[/bold cyan]")
            for idx, memory in enumerate(memories.get("1", []), 1):
                rprint(f"[green]Memory #{idx}:[/green]")
                rprint(json.dumps(memory, indent=2))
            rprint("[cyan]========================[/cyan]\n")

    def save_memory_with_timestamp(self, memory_item):
        """Save memory with additional metadata"""
        try:
            with open(LOCAL_MEMORY_FILE, 'r') as f:
                current_memories = json.load(f)
            
            timestamped_memory = {
                "content": memory_item,
                "timestamp": datetime.datetime.now().isoformat(),
                "session_id": str(uuid.uuid4())[:8]
            }
            
            current_memories["1"].append(timestamped_memory)
            
            with open(LOCAL_MEMORY_FILE, 'w') as f:
                json.dump(current_memories, f, indent=4)
            
            self.memory_stats["successful_saves"] += 1
            self.current_session_memories.append(timestamped_memory)
            return True
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            self.memory_stats["failed_saves"] += 1
            return False

    def verify_memory_storage(self, memory_item):
        """Verify if a memory was properly stored"""
        try:
            # Check local file storage
            with open(LOCAL_MEMORY_FILE, 'r') as f:
                stored_memory = json.load(f)
                if memory_item in stored_memory["1"]:
                    return True
            return False
        except Exception as e:
            logger.error(f"Memory verification failed: {e}")
            return False

    def analyze_memory(self, memory_item):
        """Analyze the quality of stored memory"""
        prompt = f"""
        Analyze this memory item and rate its quality:
        {memory_item}
        
        Consider:
        1. Completeness
        2. Structure
        3. Usefulness for future reference
        """
        
        try:
            analysis = self.test_llm.predict(prompt)
            return analysis
        except Exception as e:
            logger.error(f"Memory analysis failed: {e}")
            return "Analysis failed"

    def detect_language(self, text):
        """Simple language detection"""
        persian_chars = set('ءآأؤإئابةتثجحخدذرزسشصضطظعغفقکلمنهوىيپچژکگی')
        text_chars = set(text)
        if any(char in persian_chars for char in text_chars):
            return "fa"
        return "en"

    def extract_personal_info(self, text, lang):
        """Extract personal information from text"""
        # Define language-specific keywords
        keywords = {
            "fa": {
                "name": ["اسم", "نام"],
                "like": ["دوست دارم", "علاقه دارم"],
                "dislike": ["دوست ندارم", "متنفرم"],
                "am": ["هستم", "میباشم", "ام"]
            },
            "en": {
                "name": ["name", "called"],
                "like": ["like", "love"],
                "dislike": ["don't like", "hate", "dislike"],
                "am": ["am", "i'm", "i am"]
            }
        }

        lang_keywords = keywords.get(lang, keywords["en"])
        info = {}

        # Extract name
        if any(keyword in text.lower() for keyword in lang_keywords["name"]):
            info["type"] = "personal"
            info["category"] = "name"
            info["value"] = text

        # Extract preferences
        elif any(keyword in text.lower() for keyword in lang_keywords["dislike"]):
            info["type"] = "preference"
            info["category"] = "dislike"
            info["value"] = text

        # Extract characteristics
        elif any(keyword in text.lower() for keyword in lang_keywords["am"]):
            info["type"] = "characteristic"
            info["value"] = text

        if info:
            info["timestamp"] = datetime.datetime.now().isoformat()
            self.user_profile["facts"].append(info)
            return info
        return None

    def get_user_profile(self):
        """Retrieve and format user profile information"""
        profile_summary = {
            "en": "Based on our conversation, I know that:",
            "fa": ":براساس مکالمه‌مان، من می‌دانم که"
        }

        facts = []
        for fact in self.user_profile["facts"]:
            if fact["type"] == "personal":
                facts.append(f"- {fact['value']}")
            elif fact["type"] == "preference":
                facts.append(f"- {fact['value']}")
            elif fact["type"] == "characteristic":
                facts.append(f"- {fact['value']}")

        lang = self.user_profile.get("language", "en")
        return f"{profile_summary[lang]}\n" + "\n".join(facts)

    def run_chat_session(self):
        """Run an interactive chat session with enhanced memory display"""
        rprint("[bold green]Starting terminal chat session.[/bold green]")
        rprint("""
[bold yellow]Available commands:[/bold yellow]
- 'exit': End the session
- 'stats': Show memory statistics
- 'show memories': Display all stored memories
- 'save this: <your_info>': Save new information
- 'clear screen': Clear terminal display
        """)
        
        while True:
            try:
                user_input = input("\n[bold white][You][/bold white]: ")
                
                # Detect language
                lang = self.detect_language(user_input)
                self.user_profile["language"] = lang

                if user_input.lower() == 'exit':
                    self.display_memory_stats()
                    rprint("[bold red]Ending chat session[/bold red]")
                    break
                
                elif user_input.lower() == 'stats':
                    self.display_memory_stats()
                    continue
                
                elif user_input.lower() == 'show memories':
                    self.display_current_memories()
                    continue
                
                elif user_input.lower() == 'clear screen':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue

                elif user_input.lower() in ['profile', 'what you know about me']:
                    profile = self.get_user_profile()
                    rprint(f"[bold cyan][Assistant]:[/bold cyan] {profile}")
                    continue

                # Process the message using existing workflow
                state = {
                    "user_input": user_input,
                    "conversation_history": [],
                    "extracted_memory": None
                }
                
                result = process_message(state)
                
                # Display the response
                rprint(f"[bold cyan][Assistant]:[/bold cyan] {result['user_input']}")
                
                # Handle memory extraction and storage
                if "save this:" in user_input.lower():
                    try:
                        # Extract the content after "save this:"
                        content_to_save = user_input.split("save this:", 1)[1].strip()
                        memory_item = extract_data(content_to_save)
                        
                        rprint("[bold yellow]Processing memory...[/bold yellow]")
                        
                        if self.save_memory_with_timestamp(memory_item):
                            self.memory_stats["total_memories"] += 1
                            rprint("[bold green]✓ Memory successfully stored[/bold green]")
                            
                            # Show the stored memory
                            rprint("[bold blue]Stored Memory:[/bold blue]")
                            rprint(json.dumps(memory_item, indent=2))
                            
                            # Analyze quality
                            analysis = self.analyze_memory(memory_item)
                            rprint(f"[bold magenta]Memory Analysis:[/bold magenta]\n{analysis}")
                        else:
                            rprint("[bold red]× Failed to store memory[/bold red]")
                    
                    except Exception as e:
                        logger.error(f"Error processing memory: {e}")
                        rprint(f"[bold red]Error: {e}[/bold red]")

                # Process the message and extract personal information
                extracted_info = self.extract_personal_info(user_input, lang)
                if extracted_info:
                    rprint("[bold yellow]New information extracted and stored![/bold yellow]")

            except KeyboardInterrupt:
                rprint("\n[bold red]Chat session interrupted[/bold red]")
                self.display_memory_stats()
                break
            except Exception as e:
                logger.error(f"Error in chat session: {e}")
                rprint(f"[bold red]Error: {e}[/bold red]")

if __name__ == "__main__":
    # Initialize and run the terminal chat tester
    chat_tester = TerminalChatTester()
    chat_tester.run_chat_session()