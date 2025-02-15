import os
import datetime
import logging
import threading
import atexit
import time
import json
import requests
import telebot
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.chains.summarize import load_summarize_chain
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from utils.helpers import escape_markdown_v2, refine_ai_response

# New imports for conversation memory
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

# Configure logging to display INFO level messages on terminal.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import warnings
# Suppress deprecation warnings from langchain memory (adjust as needed per migration guide)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Telegram API Token and MongoDB Config
TELEGRAM_BOT_TOKEN = '8028667097:AAEOQqzrC9r14j1BLF2fWTuh1ZcKpItzFEA'
MONGO_CONNECTION_STRING = "mongodb://localhost:27017"
DATABASE_NAME = "user_data"
COLLECTION_NAME = "chat_history"
BUSINESS_INFO_COLLECTION = "business_info"

# LangChain Setup
llm = ChatOpenAI(
    base_url="http://localhost:15203/v1",  # G4F API server URL
    model_name="gpt-4o-mini",  # Model name for chat completions
    temperature=0.5
)
logging.info("LangChain LLM initialized for single-user chat.")

# MongoDB Helper: Retrieve the user's chat history for summarization
def get_history_for_chat(telegram_chat_id: str):
    session_id = f"{telegram_chat_id}_{int(datetime.datetime.now().timestamp())}"
    history_obj = MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGO_CONNECTION_STRING,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME,
    )
    return history_obj

# MongoDB Helper: Save messages to chat history
def save_message_to_history(chat_id, role, content):
    try:
        history_obj = get_history_for_chat(chat_id)
        if role == "user":
            message_obj = HumanMessage(content=content)
        elif role == "assistant":
            message_obj = AIMessage(content=content)
        else:
            message_obj = HumanMessage(content=content)
        history_obj.add_message(message_obj)
        logging.info(f"Saved message to history for chat '{chat_id}': {content[:50]}")
    except Exception as e:
        logging.error(f"Error saving message to history for chat '{chat_id}': {e}")

# LangChain Prompt Setup (Single User)
prompt_template_text = "Hello! How can I assist you today?"
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt_template_text),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("User Input: {input}\nAI Tone: {ai_tone}")
])

# Create the LangChain chain for generating responses
chain = prompt | llm
logging.info("LangChain chain for single-user chat initialized.")

# Configure the chain with message history for single-user interactions
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda telegram_chat_id: get_history_for_chat(telegram_chat_id),
    input_messages_key="input",
    history_messages_key="history",
)
logging.info("LangChain chain with message history configured for single-user chat.")

# Function to get summarized history for a single user session
def get_summarized_history_for_session(session_id: str) -> str:
    history_obj = MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGO_CONNECTION_STRING,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME,
    )
    messages = history_obj.messages
    if not messages:
        return "No messages."
    
    combined = "\n".join([msg.content if hasattr(msg, "content") else str(msg) for msg in messages])
    try:
        summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = summary_chain.run(combined)
    except Exception as e:
        summary = f"Summary failed: {str(e)}"
    return summary

# MongoDB Helper Functions
def get_mongo_collection():
    try:
        client = MongoClient(MONGO_CONNECTION_STRING)
        db = client[DATABASE_NAME]
        logging.info("Connected to MongoDB (database: '%s').", DATABASE_NAME)
        return db[COLLECTION_NAME]
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {str(e)}")
        raise e

def get_user_business_info(chat_id: str) -> str:
    collection = get_mongo_collection()
    result = collection.find_one({"chat_id": chat_id})
    return result["business_info"] if result else ""

def save_user_business_info(chat_id: str, info: str):
    collection = get_mongo_collection()
    collection.update_one(
        {"chat_id": chat_id},
        {"$set": {"business_info": info, "updated_at": datetime.datetime.utcnow()}},
        upsert=True
    )
    logging.info(f"Updated business info for chat '{chat_id}'")

# Telegram Bot Initialization
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
bot.set_my_commands([
    telebot.types.BotCommand("newchat", "Start a new chat session"),
    telebot.types.BotCommand("summary", "Get chat summary"),
    telebot.types.BotCommand("settings", "Modify bot settings")
])
logging.info("Bot commands registered: newchat, summary, settings")

# AI Tone map to store user preferences
ai_tone_map = {}

# Global dictionary to store per-session memory (buffer and summary)
session_memory = {}

# Handling user text messages with conversation memory integration
@bot.message_handler(func=lambda message: message.text is not None and not message.text.startswith("/"))
def handle_single_user_message(message):
    chat_id = str(message.chat.id)
    user_message_text = message.text
    sender_first_name = message.from_user.first_name or message.from_user.username
    logging.info(f"Received message from '{sender_first_name}' (chat {chat_id}): {user_message_text}")
    
    # Save user message to MongoDB history
    save_message_to_history(chat_id, "user", user_message_text)
    logging.info("User message saved to MongoDB.")
    
    # Set up conversation memory if not yet created
    if chat_id not in session_memory:
        buffer_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        summary_memory = ConversationSummaryMemory(llm=llm, memory_key="chat_summary", input_key="input", output_key="output")
        session_memory[chat_id] = {"buffer": buffer_memory, "summary": summary_memory}
        logging.info(f"Initialized conversation memory for chat '{chat_id}'.")
    
    # Load recent conversation from the buffer memory only
    memory_vars = session_memory[chat_id]["buffer"].load_memory_variables({})
    chat_history = memory_vars.get("chat_history", [])
    logging.info(f"Loaded {len(chat_history)} messages from conversation memory for chat '{chat_id}'.")
    
    prompt_input = f"{sender_first_name}: {user_message_text}"
    logging.info(f"Preparing prompt with input: {prompt_input}, AI tone: {ai_tone_map.get(chat_id, 'friendly')}")
    
    # Format prompt using conversation memory
    full_prompt = prompt.format(input=prompt_input, ai_tone=ai_tone_map.get(chat_id, "friendly"), history=chat_history)
    logging.info("Formatted prompt: " + full_prompt[:100] + " ...")
    
    try:
        # Invoke chain with context and log raw response
        logging.info("Invoking LangChain chain with the prompt and history...")
        ai_response = chain.invoke({
            "input": prompt_input,
            "ai_tone": ai_tone_map.get(chat_id, "friendly"),
            "history": chat_history
        })
        logging.info(f"Chain invocation successful. Raw response: {ai_response}")
        
        # Refine the AI response using helper functions for proper Telegram output
        unrefined_response = ai_response.content
        refined_response = refine_ai_response(unrefined_response)
        refined_response = escape_markdown_v2(refined_response)
        logging.info("AI response refined for Telegram UI.")
        
        # Save assistant's reply to MongoDB
        save_message_to_history(chat_id, "assistant", refined_response)
        logging.info("Assistant response saved to MongoDB.")
        
        # Update conversation memory with the new interaction
        session_memory[chat_id]["buffer"].save_context({"input": prompt_input}, {"output": refined_response})
        logging.info("Updated conversation memory (buffer) with new interaction.")
        
        # Optionally: update incremental summary if conversation is long
        if len(chat_history) > 5:
            session_memory[chat_id]["summary"].save_context({"input": prompt_input}, {"output": refined_response})
            logging.info("Updated conversation summary memory.")
        
        # Send refined response to user
        bot.send_message(chat_id, refined_response, parse_mode="MarkdownV2")
        logging.info(f"Sent refined response to chat '{chat_id}'.")
    except Exception as e:
        error_message = "❌ Something went wrong while processing your message."
        bot.send_message(chat_id, error_message)
        logging.error(f"Error processing message for chat '{chat_id}': {e}")

# Send summary on command
@bot.message_handler(commands=['summary'])
def send_summary(message):
    chat_id = str(message.chat.id)
    
    # Create the session_id based on chat_id
    session_id = f"{chat_id}_{int(datetime.datetime.now().timestamp())}"
    
    # Get the summarized history for the session
    summary = get_summarized_history_for_session(session_id)
    
    # Send the summary back to the user
    bot.send_message(chat_id, summary)

# Set up bot commands for AI tone and business info
@bot.message_handler(commands=['settings'])
def settings(message):
    chat_id = str(message.chat.id)
    
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_ai_tone = telebot.types.InlineKeyboardButton("Select AI Tone", callback_data="ai_tone")
    btn_business_info = telebot.types.InlineKeyboardButton("Set Business Info", callback_data="set_business_info")
    keyboard.add(btn_ai_tone, btn_business_info)
    
    settings_text = "⚙️ *Settings*\nPlease choose an option to configure."
    bot.send_message(chat_id, settings_text, reply_markup=keyboard, parse_mode="Markdown")

# Handle button presses for AI tone selection
@bot.callback_query_handler(func=lambda call: call.data == "ai_tone")
def handle_ai_tone(call):
    chat_id = str(call.message.chat.id)
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_friendly = telebot.types.InlineKeyboardButton("Friendly", callback_data="ai_tone_friendly")
    btn_formal = telebot.types.InlineKeyboardButton("Formal", callback_data="ai_tone_formal")
    btn_professional = telebot.types.InlineKeyboardButton("Professional", callback_data="ai_tone_professional")
    keyboard.add(btn_friendly, btn_formal, btn_professional)
    bot.send_message(chat_id, "Please select an AI tone:", reply_markup=keyboard)

# Set the AI tone based on user's selection
@bot.callback_query_handler(func=lambda call: call.data.startswith("ai_tone_"))
def set_ai_tone(call):
    chat_id = str(call.message.chat.id)
    tone_map = {
        "ai_tone_friendly": "Friendly",
        "ai_tone_formal": "Formal",
        "ai_tone_professional": "Professional"
    }
    selected_tone = tone_map.get(call.data)
    ai_tone_map[chat_id] = selected_tone
    bot.send_message(chat_id, f"AI tone set to {selected_tone}.")
    bot.answer_callback_query(call.id)

# Handle button presses for Business Info setting
@bot.callback_query_handler(func=lambda call: call.data == "set_business_info")
def handle_business_info(call):
    chat_id = str(call.message.chat.id)
    bot.send_message(chat_id, "Please send your business information:")

# Handle user text to save business info
@bot.message_handler(func=lambda message: message.text and message.chat.id in ai_tone_map)
def save_business_info(message):
    chat_id = str(message.chat.id)
    business_info = message.text
    bot.send_message(chat_id, f"Business info saved: {business_info}")
    logging.info(f"Business info saved for chat '{chat_id}': {business_info}")

@bot.message_handler(commands=['newchat'])
def new_chat(message):
    chat_id = str(message.chat.id)
    session_memory.pop(chat_id, None)
    bot.send_message(chat_id, "🔄 New chat session started. Conversation memory cleared.")
    logging.info(f"New chat session started for chat '{chat_id}'.")

# Run the bot in a separate thread to avoid blocking
def run_bot():
    bot.polling(none_stop=True, timeout=30)

# Running bot in a separate thread
bot_thread = threading.Thread(target=run_bot)
bot_thread.daemon = True
bot_thread.start()

logging.info("Bot is running in a separate thread.")

# Wait for the bot thread to finish to keep the main thread alive.
bot_thread.join()
