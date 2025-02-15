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
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
import warnings

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----- Configuration & Global Dictionaries -----
TELEGRAM_BOT_TOKEN = '8028667097:AAEOQqzrC9r14j1BLF2fWTuh1ZcKpItzFEA'
MONGO_CONNECTION_STRING = "mongodb://localhost:27017"
DATABASE_NAME = "user_data"
COLLECTION_NAME = "chat_history"

# Global dictionaries for user settings
ai_tone_map = {}
personal_infos = {} 
personal_info_update_pending = {}

# NEW: Add globals for business info handling
business_info_update_pending = {}
business_info_mode = {}

session_memory = {}

# ----- LangChain Setup -----
llm = ChatOpenAI(
    base_url="http://localhost:15203/v1",
    model_name="gpt-4o-mini",
    api_key="324",
    temperature=0.5
)
logging.info("LangChain LLM initialized.")

prompt_template_text = (
    "اطلاعات شخصی کاربر: {personal_info}\n"
    "سلام! چطور می‌توانم به شما کمک کنم؟"
)
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt_template_text),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("ورودی کاربر: {input}\nلحن هوش مصنوعی: {ai_tone}")
])
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda telegram_chat_id: get_history_for_chat(telegram_chat_id),
    input_messages_key="input",
    history_messages_key="history",
)
logging.info("LangChain chain configured.")

# ----- Helper Functions -----
def safe_send_message(chat_id, text, parse_mode="MarkdownV2", reply_markup=None):
    try:
        bot.send_message(chat_id, escape_markdown_v2(text), parse_mode=parse_mode, reply_markup=reply_markup)
    except Exception as e:
        logging.error("Failed sending message to chat '%s': %s", chat_id, e)

def get_history_for_chat(telegram_chat_id: str):
    session_id = f"{telegram_chat_id}_{int(datetime.datetime.now().timestamp())}"
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGO_CONNECTION_STRING,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME,
    )

def save_message_to_history(chat_id, role, content):
    try:
        history_obj = get_history_for_chat(chat_id)
        message_obj = HumanMessage(content=content) if role != "assistant" else AIMessage(content=content)
        history_obj.add_message(message_obj)
        logging.info("Saved message for chat '%s': %s", chat_id, content[:50])
    except Exception as e:
        logging.error("Error saving message for chat '%s': %s", chat_id, e)

def get_mongo_collection():
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DATABASE_NAME]
    logging.info("Connected to MongoDB (db: '%s').", DATABASE_NAME)
    return db[COLLECTION_NAME]

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
    logging.info("Updated business info for chat '%s'", chat_id)

# ----- Telegram Bot Initialization -----
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
bot.set_my_commands([
    telebot.types.BotCommand("newchat", "Start a new chat session"),
    telebot.types.BotCommand("summary", "Get chat summary"),
    telebot.types.BotCommand("settings", "Modify bot settings")
])
logging.info("Bot commands registered.")

# ----- Message Handlers & Callbacks -----
@bot.message_handler(func=lambda msg: msg.text and not msg.text.startswith("/"))
def handle_single_user_message(message):
    chat_id = str(message.chat.id)
    # If user is updating personal info, delegate to that handler.
    if chat_id in personal_info_update_pending:
        return
    user_text = message.text
    sender = message.from_user.first_name or message.from_user.username
    logging.info("Message from '%s' (chat %s): %s", sender, chat_id, user_text)
    save_message_to_history(chat_id, "user", user_text)
    
    if chat_id not in session_memory:
        session_memory[chat_id] = {
            "buffer": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            "summary": ConversationSummaryMemory(llm=llm, memory_key="chat_summary", input_key="input", output_key="output")
        }
        logging.info("Initialized conversation memory for chat '%s'.", chat_id)
        
    chat_history = session_memory[chat_id]["buffer"].load_memory_variables({}).get("chat_history", [])
    prompt_input = f"{sender}: {user_text}"
    full_prompt = prompt.format(
        input=prompt_input,
        ai_tone=ai_tone_map.get(chat_id, "friendly"),
        history=chat_history,
        personal_info=personal_infos.get(chat_id, "—")
    )
    logging.info("Formatted prompt (first 100 chars): %s ...", full_prompt[:100])
    
    try:
        # Pass the "personal_info" variable to the chain
        ai_response = chain.invoke({
            "input": prompt_input,
            "ai_tone": ai_tone_map.get(chat_id, "friendly"),
            "history": chat_history,
            "personal_info": personal_infos.get(chat_id, "—")
        })
        refined = escape_markdown_v2(refine_ai_response(ai_response.content))
        save_message_to_history(chat_id, "assistant", refined)
        session_memory[chat_id]["buffer"].save_context({"input": prompt_input}, {"output": refined})
        if len(chat_history) > 5:
            session_memory[chat_id]["summary"].save_context({"input": prompt_input}, {"output": refined})
        safe_send_message(chat_id, refined)
        logging.info("Response sent to chat '%s'.", chat_id)
    except Exception as e:
        safe_send_message(chat_id, "❌ Something went wrong while processing your message.")
        logging.error("Error processing message for chat '%s': %s", chat_id, e)

@bot.message_handler(func=lambda m: m.text and str(m.chat.id) in personal_info_update_pending)
def update_personal_info(message):
    chat_id = str(message.chat.id)
    personal_infos[chat_id] = message.text
    del personal_info_update_pending[chat_id]
    safe_send_message(chat_id, f"✅ اطلاعات شخصی شما به‌روزرسانی شد: {message.text}")
    logging.info("Personal info updated for chat '%s'.", chat_id)

@bot.message_handler(commands=['summary'])
def send_summary(message):
    chat_id = str(message.chat.id)
    session_id = f"{chat_id}_{int(datetime.datetime.now().timestamp())}"
    summary = get_summarized_history_for_session(session_id)
    safe_send_message(chat_id, summary)

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
        summary = load_summarize_chain(llm, chain_type="map_reduce").run(combined)
    except Exception as e:
        summary = f"Summary failed: {str(e)}"
    return summary

@bot.message_handler(commands=['settings'])
def bot_settings(message):
    chat_id = str(message.chat.id)
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_personal_info = telebot.types.InlineKeyboardButton("تنظیم اطلاعات شخصی", callback_data="set_personal_info")
    btn_ai_tone = telebot.types.InlineKeyboardButton("انتخاب لحن هوش مصنوعی", callback_data="ai_tone")
    keyboard.add(btn_personal_info, btn_ai_tone)
    settings_text = "⚙️ *تنظیمات ربات:*\n\nلطفاً گزینه‌ای را انتخاب کنید:"
    bot.reply_to(message, escape_markdown_v2(settings_text), reply_markup=keyboard, parse_mode="MarkdownV2")
    save_message_to_history(chat_id, "system", settings_text)

@bot.callback_query_handler(func=lambda call: call.data == "set_personal_info")
def handle_set_personal_info(call):
    chat_id = str(call.message.chat.id)
    personal_info_update_pending[chat_id] = True
    bot.answer_callback_query(call.id, "لطفاً اطلاعات شخصی خود را ارسال کنید.")
    safe_send_message(chat_id, "📄 لطفاً اطلاعات شخصی خود را به عنوان پیام ارسال کنید:")

@bot.callback_query_handler(func=lambda call: call.data == "ai_tone")
def handle_ai_tone(call):
    chat_id = str(call.message.chat.id)
    keyboard = telebot.types.InlineKeyboardMarkup()
    tones = [
        ("دوستانه", "ai_tone_dostane"),
        ("رسمی", "ai_tone_rasmi"),
        ("حرفه ای", "ai_tone_pro")
    ]
    for label, callback_data in tones:
        keyboard.add(telebot.types.InlineKeyboardButton(label, callback_data=callback_data))
    bot.answer_callback_query(call.id)
    safe_send_message(chat_id, "لطفاً یکی از لحن‌های زیر را انتخاب کنید:", reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: call.data in ["ai_tone_dostane", "ai_tone_rasmi", "ai_tone_pro"])
def select_ai_tone(call):
    chat_id = str(call.message.chat.id)
    mapping = {
        "ai_tone_dostane": ("دوستانه", "friendly, cool and kind"),
        "ai_tone_rasmi": ("رسمی", "official, serious and formal"),
        "ai_tone_pro": ("حرفه ای", "professional, expert and business-like")
    }
    selected = mapping[call.data]
    ai_tone_map[chat_id] = f"{selected[0]}: {selected[1]}"
    bot.answer_callback_query(call.id, f"لحن انتخاب شده: {selected[0]}")
    safe_send_message(chat_id, f"✅ لحن هوش مصنوعی به '{selected[0]}' تغییر یافت.")

# Additional handlers (for business info, /newchat, /start, /help, /about, /options, etc.)
# ...existing code for business info callbacks and commands...
@bot.message_handler(commands=['newchat'])
def new_chat(message):
    chat_id = str(message.chat.id)
    session_memory.pop(chat_id, None)
    safe_send_message(chat_id, "🔄 New chat session started. Conversation memory cleared.")
    logging.info("New chat session started for chat '%s'.")

@bot.message_handler(commands=['start'])
def send_welcome(message):
    chat_id = str(message.chat.id)
    welcome_msg = "🤖 سلام! به ربات چت خوش آمدید! امروز چگونه می‌توانم به شما کمک کنم؟"
    bot.reply_to(message, escape_markdown_v2(welcome_msg), parse_mode="MarkdownV2")
    save_message_to_history(chat_id, "system", welcome_msg)

@bot.message_handler(commands=['help'])
def send_help(message):
    chat_id = str(message.chat.id)
    help_msg = (
        "ℹ️ *راهنمای ربات:*\n"
        "• /start - شروع مجدد و نمایش پیام خوش آمدگویی\n"
        "• /newchat - جلسه چت جدید\n"
        "• /summary - دریافت خلاصه گفتگو\n"
        "• /settings - تغییر تنظیمات ربات\n"
        "• /about - درباره ربات"
    )
    bot.reply_to(message, escape_markdown_v2(help_msg), parse_mode="MarkdownV2")
    save_message_to_history(chat_id, "system", help_msg)

@bot.message_handler(commands=['about'])
def about_bot(message):
    chat_id = str(message.chat.id)
    about_text = (
        "🤖 *درباره ربات:*\n"
        "ربات چت خصوصی پشتیبانی‌شده توسط LangChain و OpenAI، با امکاناتی برای درخواست‌های روزانه."
    )
    bot.reply_to(message, escape_markdown_v2(about_text), parse_mode="MarkdownV2")
    save_message_to_history(chat_id, "system", about_text)

@bot.message_handler(commands=['options'])
def options_handler(message):
    chat_id = str(message.chat.id)
    keyboard = telebot.types.InlineKeyboardMarkup()
    options = [
        ("Daily Tasks", "daily_tasks"),
        ("Instagram Story Idea", "instagram_story_idea"),
        ("Chat Summary Report", "chat_report")
    ]
    for label, callback in options:
        keyboard.add(telebot.types.InlineKeyboardButton(label, callback_data=callback))
    options_text = "⚙️ *Select an Option:*\nPlease choose one of the following:"
    bot.reply_to(message, escape_markdown_v2(options_text), reply_markup=keyboard, parse_mode="MarkdownV2")
    save_message_to_history(chat_id, "system", options_text)

@bot.callback_query_handler(func=lambda call: call.data in ["daily_tasks", "instagram_story_idea", "chat_report"])
def handle_options(call):
    chat_id = str(call.message.chat.id)
    prompts = {
        "daily_tasks": "لطفاً جزئیات وظایف روزانه خود را ارسال کنید.",
        "instagram_story_idea": "یک ایده برای داستان اینستاگرام به من بده.",
        "chat_report": "یک گزارش خلاصه گفتگو تولید کن."
    }
    prompt_input = prompts[call.data]
    ai_tone = ai_tone_map.get(chat_id, "دوستانه")
    bot.send_chat_action(chat_id, 'typing')
    placeholder = bot.send_message(chat_id, escape_markdown_v2(f"⏳ در حال پردازش {call.data.replace('_', ' ')}..."), parse_mode="MarkdownV2")
    try:
        ai_response = chain.invoke({"input": prompt_input, "ai_tone": ai_tone, "history": []})
        refined = escape_markdown_v2(refine_ai_response(ai_response.content))
        save_message_to_history(chat_id, "assistant", refined)
        bot.edit_message_text(refined, chat_id=chat_id, message_id=placeholder.message_id, parse_mode="MarkdownV2")
    except Exception as e:
        error_msg = "❌ خطا در پردازش درخواست."
        bot.edit_message_text(error_msg, chat_id=chat_id, message_id=placeholder.message_id, parse_mode="MarkdownV2")
        logging.error("Error in option '%s' for chat '%s': %s", call.data, chat_id, e)

# ----- Bot Runner -----
def run_bot():
    bot.polling(none_stop=True, timeout=30)

bot_thread = threading.Thread(target=run_bot)
bot_thread.daemon = True
bot_thread.start()
logging.info("Bot is running in a separate thread.")
bot_thread.join()
