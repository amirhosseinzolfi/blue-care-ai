from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import logging

# --- Configuration (same as full agent) ---
MODEL_NAME = "gpt-4o-mini"
BASE_URL = "http://localhost:15203/v1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatOpenAI(
    base_url=BASE_URL,
    model_name=MODEL_NAME,
    api_key="324",
    temperature=0.5
)

def terminal_chat():
    conversation = []
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

if __name__ == "__main__":
    terminal_chat()
