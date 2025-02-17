import os
import getpass
from typing import Literal
import uuid
import logging
from datetime import datetime
from rich.panel import Panel
from rich.table import Table
from rich import box
from time import time
import json
import os.path
import re

import telebot
from rich.console import Console
from telebot.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BotCommand,
    BotCommandScopeDefault,
    BotCommandScopeAllGroupChats
)
from telebot.apihelper import ApiTelegramException

# --- LangChain / LangGraph Imports ---
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.store.memory import InMemoryStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from ai_prompts import SYSTEM_PROMPT  # (Not used directly anymore)

# --- Constants ---
OPTIMIZED_SYSTEM_PROMPT = (
    "You are a digital kids nurturing assistant named Blue. Provide accurate, personalized, and natural advice to parents regarding child care. "
    "Consider the following child information: {kids_information}. "
    "Also, take into account the previous conversation history: {conversation_context}. "
    "Adjust your tone based on this setting: {ai_tone}. "
    "Do not include any greetings, salutations, or sticker instructions in your response. "
    "Be direct, concise, and fully answer the user's query in warm, supportive Persian."
)
STICKER_ID = "CAACAgIAAxkBAAEB8G9gJ-7jFi7aPBSm8M7pJBrqrU_QAAJFAAMvJIOUwiuEvjYaq7sE"
STICKER_START = "CAACAgIAAxkBAAEB8IxgJ-start-sticker"
STICKER_HELP = "CAACAgIAAxkBAAEB8JxgJ-help-sticker"
STICKER_SETTING = None
USER_DATA_BASE_PATH = "user_data"
DEFAULT_AI_TONE = "دوستانه"
LOADING_MESSAGE_TEXT = "درحال فکر کردن 🧐..."
MARKDOWN_V2_PARSE_MODE = "MarkdownV2"
JSON_FILE_ENCODING = "utf-8"

# --- Global variable for settings state ---
setting_data = {}

# --- Telegram Bot Setup ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8028667097:AAEOQqzrC9r14j1BLF2fWTuh1ZcKpItzFEA")
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# --- Rich Console & Logging Setup ---
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Environment Variable Setup ---
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
        logger.info(f"Environment variable '{var}' set by user input.")
    else:
        logger.info(f"Environment variable '{var}' already set.")

_set_env("OPENAI_API_KEY")

# --- LangChain / LangGraph Setup ---
llm = ChatOpenAI(
    base_url="http://localhost:15203/v1",
    model_name="gemini-1.5-flash",
    temperature=0.5,
    streaming=True
)
logger.info("ChatOpenAI LLM initialized with streaming enabled.")

# Initialize memory, embeddings, and in-memory store
memory = MemorySaver()
logger.info("MemorySaver initialized.")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
logger.info("OllamaEmbeddings initialized.")

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 768,
        "fields": ["$"]
    }
)
logger.info("InMemoryStore initialized.")

# --- Removed get_weather Tool (no longer used) ---

# --- Agent Graph Creation ---
graph = create_react_agent(llm, tools=[], checkpointer=memory, store=store)
logger.info("ReAct agent graph created.")

# --- Utility Functions ---
def refine_ai_response(text: str) -> str:
    text = re.sub(r"\[Sticker:.*?\]", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"*\1*", text)
    text = re.sub(r"__([^_]+)__", r"*\1*", text)
    text = re.sub(r"^####\s+(.*?)$", r"🔶 \1", text, flags=re.MULTILINE)
    text = re.sub(r"^###\s+(.*?)$", r"⭐ \1", text, flags=re.MULTILINE)
    text = re.sub(r"^##\s+(.*?)$", r"🔷 \1", text, flags=re.MULTILINE)
    text = re.sub(r"^#\s+(.*?)$", r"🟣 \1", text, flags=re.MULTILINE)
    text = re.sub(r"^(?:\s*[-*]\s+)(.*?)$", r"🔹 \1", text, flags=re.MULTILINE)
    text = re.sub(r"^(?:\s*\d+\.\s+)(.*?)$", r"🔹 \1", text, flags=re.MULTILINE)
    return text.strip()

def escape_markdown_v2(text: str) -> str:
    return re.sub(r"([_*[\]()~`>#+\-=|{}.!])", r"\\\1", text)

# --- Kid Info Analysis ---
llm_for_kid_info = ChatOpenAI(
    base_url="http://localhost:15203/v1",
    model_name="gemini-1.5-flash",
    temperature=0.3
)
logger.info("ChatOpenAI for kid info analysis initialized.")

def analyze_and_structure_kid_info(new_info: str, old_info: str = "") -> str:
    combined = old_info + "\n" + new_info if old_info else new_info
    prompt = (
        "You are a Persian assistant. Take the user's child info below, then clean, "
        "organize, and structure it in an attractive, well-formatted Persian text."
        "\n\nChild info:\n" + combined
    )
    try:
        response = llm_for_kid_info.invoke(prompt)
        result = response.content.strip()
        logger.info("Kid info analysis completed successfully.")
        return result
    except Exception as e:
        logger.exception("Error during kid info analysis:")
        return new_info

# --- Bot Logger Class ---
class BotLogger:
    def __init__(self):
        self.console = Console()
        self.start_time = time()
        logger.info("BotLogger initialized.")

    def log_stage(self, stage: str, message: str, style: str = "blue"):
        elapsed = time() - self.start_time
        panel = Panel(
            f"[bold {style}]{message}[/bold {style}]",
            title=f"[{elapsed:.2f}s] {stage}",
            border_style=style,
            box=box.ROUNDED
        )
        self.console.print(panel)
        logger.info(f"Stage: {stage} - {message}")

    def log_memory(self, memories: list, user_id: str):
        table = Table(title=f"Memories for User {user_id}", box=box.DOUBLE_EDGE)
        table.add_column("Time", style="cyan")
        table.add_column("Memory", style="green")
        for mem in memories:
            timestamp = datetime.now().strftime("%H:%M:%S")
            table.add_row(timestamp, str(mem.value))
        self.console.print(table)
        logger.info(f"Memories logged for user_id: {user_id}")

    def log_error(self, error: str):
        self.console.print(f"[bold red]ERROR:[/bold red] {error}", style="red")
        logger.error(f"ERROR: {error}")

bot_logger = BotLogger()

# --- User Memory Class ---
class UserMemory:
    def __init__(self, user_id: str, base_path: str = USER_DATA_BASE_PATH):
        self.user_id = user_id
        self.base_path = base_path
        self.memory_file = os.path.join(base_path, f"{user_id}_memory.json")
        self.history_file = os.path.join(base_path, f"{user_id}_history.json")
        self.kid_info_file = os.path.join(base_path, f"{user_id}_kid_info.json")
        self.ai_tone_file = os.path.join(base_path, f"{user_id}_ai_tone.json")
        self._ensure_files_exist()
        logger.info(f"UserMemory initialized for user_id: {user_id}")

    def _ensure_files_exist(self):
        os.makedirs(self.base_path, exist_ok=True)
        for file in [self.memory_file, self.history_file, self.kid_info_file, self.ai_tone_file]:
            if not os.path.exists(file):
                with open(file, "w", encoding=JSON_FILE_ENCODING) as f:
                    json.dump([], f)
        logger.info(f"Ensured existence of user data files for user_id: {self.user_id}")

    def refresh_memory(self):
        for file in [self.memory_file, self.history_file, self.kid_info_file, self.ai_tone_file]:
            with open(file, "w", encoding=JSON_FILE_ENCODING) as f:
                json.dump([], f)
        logger.info(f"Cleared file-based memory for user {self.user_id}")

    def get_kid_info_data(self):
        return self._load_json_data(self.kid_info_file)

    def update_kid_info(self, new_info: str, mode: str = "replace"):
        return self._update_json_data(self.kid_info_file, new_info, mode)

    def get_ai_tone_data(self):
        return self._load_json_data(self.ai_tone_file)

    def update_ai_tone(self, new_tone: str, mode: str = "replace"):
        return self._update_json_data(self.ai_tone_file, new_tone, mode)

    def get_memories(self):
        return self._load_json_data(self.memory_file)

    def get_history(self):
        return self._load_json_data(self.history_file)

    def add_memory(self, memory_item):
        return self._append_json_data(self.memory_file, {
            "timestamp": datetime.now().isoformat(),
            "content": memory_item
        })

    def add_to_history(self, message, role="user"):
        return self._append_json_data(self.history_file, {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": message
        })

    def get_recent_history(self, limit=10):
        history = self.get_history()
        return history[-limit:] if history else []

    def get_kid_info(self):
        info = self.get_kid_info_data()
        return info if info else "No kid information provided"

    def _load_json_data(self, file_path):
        try:
            with open(file_path, "r", encoding=JSON_FILE_ENCODING) as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}. It will be created.")
            return []
        except json.JSONDecodeError:
            logger.warning(f"JSONDecodeError loading {file_path}. File may be empty or corrupt. Initializing with empty list.")
            return []

    def _update_json_data(self, file_path, new_data, mode="replace"):
        current_data = self._load_json_data(file_path)
        if mode == "add" and isinstance(current_data, str):
            updated_data = current_data + " " + new_data
        elif mode == "add" and isinstance(current_data, list):
            updated_data = current_data + [new_data]
        else:
            updated_data = new_data
        try:
            with open(file_path, "w", encoding=JSON_FILE_ENCODING) as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Updated JSON data in {file_path}, mode: {mode}")
            return updated_data
        except Exception as e:
            logger.error(f"Error updating JSON data in {file_path}: {e}")
            return current_data

    def _append_json_data(self, file_path, new_item):
        current_data = self._load_json_data(file_path)
        if not isinstance(current_data, list):
            current_data = []
        current_data.append(new_item)
        try:
            with open(file_path, "w", encoding=JSON_FILE_ENCODING) as f:
                json.dump(current_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Appended data to {file_path}")
        except Exception as e:
            logger.error(f"Error appending data to {file_path}: {e}")

# --- Function to Clear InMemoryStore for a user ---
def clear_store_for_user(user_id: str):
    namespace = (user_id, "memories")
    search_results = store.search(namespace, query="", limit=1000)
    for item in search_results:
        try:
            store.delete(namespace, item.key)
        except Exception as e:
            logger.error(f"Error deleting memory item {getattr(item, 'key', 'unknown')}: {e}")
    logger.info(f"Cleared long term memory for user {user_id} from InMemoryStore.")

# --- Agent Execution Function ---
def run_agent(query, config, chat_id, message_id):
    bot_logger.log_stage("Query Received", f"Processing query: {query}")
    start_time = time()
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        bot_logger.log_error("user_id not provided in config")
        raise ValueError("user_id must be provided in the config for memory store usage.")

    user_memory = UserMemory(user_id)
    bot_logger.log_stage("Memory Initialization", f"Initialized memory for user: {user_id}")
    user_memory.add_to_history(query, "user")

    kid_info = user_memory.get_kid_info()
    config["kids_information"] = kid_info
    bot_logger.log_stage("Kid Info", f"Using kid information: {kid_info}")

    recent_history = user_memory.get_recent_history()
    bot_logger.log_stage("History", f"Loaded {len(recent_history)} recent messages")

    short_term_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
    namespace = (user_id, "memories")
    long_term_results = store.search(namespace, query=query, limit=3)
    long_term_context = "\n".join([f"Memory: {item.value['user_query']}" for item in long_term_results]) if long_term_results else ""
    conversation_context = "Short term history:\n" + short_term_context
    if long_term_context:
        conversation_context += "\nLong term memory:\n" + long_term_context

    latest_history = recent_history[-2:] if len(recent_history) >= 2 else recent_history
    short_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in latest_history])
    bot_logger.log_stage("History Summary", f"Latest 2 messages:\n{short_history_str}\nLong term memory count: {len(long_term_results) if long_term_results else 0}")
    if long_term_context:
        bot_logger.log_stage("Long Term Memory", f"Long term memory:\n{long_term_context}")
    
    ai_tone = user_memory.get_ai_tone_data() or DEFAULT_AI_TONE
    system_template = PromptTemplate.from_template(OPTIMIZED_SYSTEM_PROMPT)
    system_message = system_template.format(
        kids_information=kid_info,
        ai_tone=ai_tone,
        conversation_context=conversation_context
    )
    bot_logger.log_stage("System Message", "Built system message with conversation context")
    logger.info("Full system prompt used:\n" + system_message)

    loading_text_escaped = escape_markdown_v2(LOADING_MESSAGE_TEXT)
    loading_message = bot.send_message(chat_id, loading_text_escaped, parse_mode=MARKDOWN_V2_PARSE_MODE)
    loading_msg_id = loading_message.message_id
    bot_logger.log_stage("Loading Message", f"Sent loading message with id {loading_msg_id}")

    inputs = {
        "messages": [
            SystemMessage(content=system_message),
            HumanMessage(content=query)
        ]
    }
    bot_logger.log_stage("Processing", "Invoking agent synchronously")
    result = graph.invoke(inputs, config=config)
    
    full_result_str = json.dumps(result, default=lambda o: o.__dict__, indent=2, ensure_ascii=False)
    logger.info("Full agent response:\n" + full_result_str)
    
    if isinstance(result, dict) and "messages" in result:
        final_msg = result["messages"][-1]
        ai_response_content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
    else:
        ai_response_content = result.content if hasattr(result, "content") else str(result)
    
    if not ai_response_content:
        ai_response_content = "متأسفانه پاسخی دریافت نشد."
        bot_logger.log_error("No AI response obtained from invocation.")

    refined_response = refine_ai_response(ai_response_content)
    bot_logger.log_stage("Response Refinement", "Refined response")

    user_memory.add_to_history(refined_response, "assistant")
    user_memory.add_memory({"query": query, "response": refined_response})
    bot_logger.log_stage("History Update", "Stored AI response in history and memory")

    try:
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading_msg_id,
            text=escape_markdown_v2(refined_response),
            parse_mode=MARKDOWN_V2_PARSE_MODE
        )
    except ApiTelegramException as e:
        bot.send_message(chat_id, escape_markdown_v2(refined_response), parse_mode=MARKDOWN_V2_PARSE_MODE)

    bot_logger.log_stage("Response Sent", "AI Response Sent", "green")

    memory_id = str(uuid.uuid4())
    store.put(namespace, memory_id, {"user_query": query})
    bot_logger.log_stage("Memory Storage", f"Stored memory for user {user_id} under id {memory_id}", "yellow")

    search_results = store.search(namespace, query="", limit=5)
    if search_results:
        bot_logger.log_memory(search_results, user_id)

    elapsed_time = time() - start_time
    bot_logger.log_stage("Complete", f"Total processing time: {elapsed_time:.2f}s", "magenta")
    maybe_summarize(str(chat_id))

def maybe_summarize(thread_id: str, threshold: int = 6):
    try:
        history = memory.get_history(thread_id) if hasattr(memory, "get_history") else []
        if len(history) <= threshold:
            logger.debug(f"Conversation for thread_id: {thread_id} below threshold, no summarization.")
            return

        current_summary = memory.get_summary(thread_id) if hasattr(memory, "get_summary") else ""
        prompt_text = (
            f"Extend the summary by taking into account the new messages above:\n\n"
            f"This is summary of the conversation to date: {current_summary}"
        ) if current_summary else "Create a summary of the conversation above:"
        messages = history + [HumanMessage(content=prompt_text)]

        summarizer = ChatOpenAI(model_name="gpt-3.5-turbo-instruct")
        response = summarizer.invoke(messages)
        new_summary = response.content

        if hasattr(memory, "prune_history"):
            memory.prune_history(thread_id, keep_last=2)
        if hasattr(memory, "save_summary"):
            memory.save_summary(thread_id, new_summary)

        console.log(f"[bold cyan]Conversation summarized:[/bold cyan] {new_summary}")
        logger.info(f"Conversation summarized for thread_id: {thread_id}")

    except Exception as e:
        logger.exception(f"Error during summarization for thread_id: {thread_id}:")

# --- New Function: Refresh Bot Memory ---
def handle_refresh_memory(call):
    chat_id = call.message.chat.id
    user_id = str(chat_id)
    user_memory = UserMemory(user_id)
    user_memory.refresh_memory()
    clear_store_for_user(user_id)
    bot.send_message(chat_id, escape_markdown_v2("حافظه بات ریست شد. تمام گفتگوها و حافظه‌های قبلی پاک شدند."), parse_mode=MARKDOWN_V2_PARSE_MODE)
    logger.info(f"Refreshed all memory for user {user_id}")

# --- Telegram Bot Handlers ---
@bot.message_handler(commands=["start"])
def start_handler(message):
    reply_text = (
        "🌟 **بلو؛ دستیار هوشمند تربیتی شما!** 🌟\n\n"
        "آیا در مسیر تربیت فرزندتان با چالش‌هایی مثل لجبازی، وابستگی به گوشی، یا کمبود ایده‌های آموزشی مواجهید؟\n"
        "**بلو** اینجاست تا با راهکارهای هوشمند، مسیر تربیت رو برای شما هموار و هیجان‌انگیز کنه!\n\n"
        "✨ **ویژگی‌های منحصربه‌فرد بلو:**\n"
        "🔹 **تست شخصیت کودک:** شناخت علمی عمیق برای تربیتی بهینه\n"
        "🔹 **بازی‌های آموزشی:** جایگزینی جذاب برای فضای مجازی\n"
        "🔹 **قصه‌های آموزنده:** انتقال ارزش‌های اخلاقی و مهارتی\n"
        "🔹 **نکات تربیتی و کمک درسی:** راهکارهای کاربردی و موثر\n"
        "🔹 **کاردستی‌های خلاقانه و تصاویر شخصی‌شده:** تقویت خلاقیت و تمرکز\n\n"
        "💥 **همین حالا به جمع خانواده بلو بپیوندید و تجربه‌ای نوین از تربیت دیجیتال داشته باشید!** 🚀\n\n"
        "برای افزودن اطلاعات کودک و تغییر لحن هوش مصنوعی، از دستور /setting استفاده کنید.\n"
        "همچنین برای راهنمایی بیشتر، از دستور /help بهره ببرید."
    )
    bot.reply_to(message, escape_markdown_v2(reply_text), parse_mode=MARKDOWN_V2_PARSE_MODE)
    console.log(f"[bold blue]/start command from user {message.chat.id}[/bold blue]")
    if STICKER_START:
        bot.send_sticker(message.chat.id, STICKER_START)
        logger.info(f"Sent START sticker to user {message.chat.id}")

@bot.message_handler(commands=["help"])
def help_handler(message):
    reply_text = (
        "📘 راهنمای بات بلو:\n\n"
        "• ارسال هر پیام یعنی دریافت پاسخ هوشمندانه...\n"
        "این سیستم با بهره‌گیری از هوش مصنوعی...\n"
        "در صورت نیاز به کمک بیشتر، تماس بگیرید. 🚀"
    )
    bot.reply_to(message, reply_text)
    console.log(f"[bold blue]/help command from user {message.chat.id}[/bold blue]")
    if STICKER_HELP:
        bot.send_sticker(message.chat.id, STICKER_HELP)
        logger.info(f"Sent HELP sticker to user {message.chat.id}")

@bot.message_handler(commands=["setting"])
def setting_handler(message):
    chat_id = message.chat.id
    markup = InlineKeyboardMarkup()
    markup.add(
        InlineKeyboardButton("🎈 اطلاعات کودک", callback_data="kid_info"),
        InlineKeyboardButton("💬 لحن هوشمند", callback_data="ai_tone"),
        InlineKeyboardButton("♻️ پاکسازی حافظه", callback_data="refresh_memory")
    )
    bot.send_message(chat_id, "🔧 گزینه تنظیمات را انتخاب کنید:", reply_markup=markup)
    console.log(f"[bold blue]/setting command from user {chat_id}[/bold blue]")
    setting_data[chat_id] = {}
    if STICKER_SETTING:
        bot.send_sticker(message.chat.id, STICKER_SETTING)
        logger.info(f"Sent SETTING sticker to user {message.chat.id}")

@bot.callback_query_handler(func=lambda call: call.data in ["kid_info", "kid_info_add", "kid_info_replace", "ai_tone", "refresh_memory"])
def callback_query_handler(call):
    chat_id = call.message.chat.id
    action = call.data

    if action == "kid_info":
        handle_kid_info_callback(call)
    elif action in ["kid_info_add", "kid_info_replace"]:
        mode = action.split("_")[-1]
        message_text = (
            f"🌟 {'افزودن' if mode == 'add' else 'ثبت'} اطلاعات جدید\n\n"
            "لطفاً اطلاعات کودک خود را وارد کنید.\n"
            f"این اطلاعات به داده‌های قبلی {'اضافه خواهد شد' if mode == 'add' else 'جایگزین میشود'}."
        )
        bot.send_message(chat_id, message_text)
        setting_data[chat_id] = {"kid_info_pending": True, "kid_info_mode": mode}
        logger.info(f"Handled {action} callback for user {chat_id}")
    elif action == "ai_tone":
        handle_ai_tone_callback(call)
    elif action == "refresh_memory":
        handle_refresh_memory(call)

    bot.answer_callback_query(call.id)
    logger.info(f"Callback query handled for action: {action}, user: {chat_id}")
    
    

def handle_kid_info_callback(call):
    chat_id = call.message.chat.id
    user_id = str(chat_id)
    user_memory = UserMemory(user_id)
    current_info = user_memory.get_kid_info()

    markup = InlineKeyboardMarkup()
    markup.add(
        InlineKeyboardButton("➕ افزودن اطلاعات", callback_data="kid_info_add"),
        InlineKeyboardButton("🔄 جایگزینی اطلاعات", callback_data="kid_info_replace")
    )

    message_text = (
        "📋 اطلاعات فعلی کودک شما:\n" + current_info
        if current_info and current_info != "No kid information provided"
        else "⚠️ اطلاعاتی برای کودک ثبت نشده است.\n"
    )
    message_text += "\nچه تغییری میخواهید ایجاد کنید؟"

    bot.send_message(chat_id, message_text, reply_markup=markup)
    logger.info(f"Handled kid_info callback for user {chat_id}")

def handle_ai_tone_callback(call):
    chat_id = call.message.chat.id
    bot.send_message(chat_id, "لحن مورد نظر خود را اکنون ارسال کنید:")
    console.log(f"[bold blue]درخواست تنظیم لحن هوش مصنوعی توسط کاربر {call.from_user.id}[/bold blue]")
    setting_data[chat_id]["ai_tone_pending"] = True
    logger.info(f"Handled ai_tone callback for user {chat_id}")

@bot.message_handler(func=lambda message: True, content_types=["text"])
def handle_text_message(message):
    chat_id = message.chat.id
    user_id = str(chat_id)

    if chat_id in setting_data:
        if setting_data[chat_id].get("kid_info_pending"):
            kid_info_input = message.text
            user_memory = UserMemory(user_id)
            mode = setting_data[chat_id].get("kid_info_mode", "replace")

            waiting_message = bot.send_message(
                chat_id,
                escape_markdown_v2("درحال پردازش اطلاعات کودک... ⏳"),
                parse_mode=MARKDOWN_V2_PARSE_MODE,
            )
            waiting_message_id = waiting_message.message_id
            logger.info(f"Sent waiting message for kid info processing to user {chat_id}")

            previous_info = user_memory.get_kid_info_data() if mode == "add" else ""
            analyzed_info = analyze_and_structure_kid_info(kid_info_input, previous_info)
            refined_info = refine_ai_response(analyzed_info)
            updated_info = user_memory.update_kid_info(refined_info, mode="replace")

            confirmation_text = f"✅ اطلاعات کودک با موفقیت ثبت و ساختاردهی شد:\n{updated_info}"
            bot.edit_message_text(
                escape_markdown_v2(confirmation_text),
                chat_id=chat_id,
                message_id=waiting_message_id,
                parse_mode=MARKDOWN_V2_PARSE_MODE,
            )
            console.log(f"[bold green]Kid Information updated for user {user_id}:[/bold green] {updated_info}")
            setting_data[chat_id]["kid_info_pending"] = False
            logger.info(f"Kid info updated and setting pending flag reset for user {chat_id}")

        elif setting_data[chat_id].get("ai_tone_pending"):
            ai_tone = message.text
            user_memory = UserMemory(user_id)
            user_memory.update_ai_tone(ai_tone, mode="replace")
            reply_text = f"متوجه شدم! پاسخ‌ها به سبک '{ai_tone}' خواهند بود."
            bot.reply_to(message, reply_text)
            console.log(f"[bold green]AI Tone saved for user {user_id}:[/bold green] {ai_tone}")
            setting_data[chat_id]["ai_tone_pending"] = False
            logger.info(f"AI tone updated and setting pending flag reset for user {chat_id}")
        else:
            config = {"configurable": {"thread_id": str(chat_id), "user_id": user_id}}
            run_agent(message.text, config=config, chat_id=chat_id, message_id=message.message_id)
    else:
        config = {"configurable": {"thread_id": str(chat_id), "user_id": user_id}}
        run_agent(message.text, config=config, chat_id=chat_id, message_id=message.message_id)
    logger.info(f"Text message handled for user {chat_id}, setting_pending: {setting_data.get(chat_id)}")

def setup_bot_commands():
    bot_commands = [
        BotCommand("start", "شروع"),
        BotCommand("help", "راهنما"),
        BotCommand("setting", "تنظیمات"),
    ]
    try:
        bot.set_my_commands(commands=bot_commands, scope=BotCommandScopeDefault())
        bot.set_my_commands(commands=bot_commands, scope=BotCommandScopeAllGroupChats())
        console.log("[bold green]Bot commands set up.[/bold green]")
        logger.info("Bot commands setup completed.")
    except Exception as e:
        console.log(f"[bold red]Error setting up bot commands: {e}[/bold red]")
        logger.error(f"Error setting up bot commands: {e}")

if __name__ == "__main__":
    bot_logger.log_stage("Bot Initialization", "Starting Telegram bot polling", "green")
    setup_bot_commands()
    bot.polling(non_stop=True)
