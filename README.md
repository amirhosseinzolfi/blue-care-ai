# Blue Care AI Telegram Bot

A Telegram bot powered by LangChain and an LLM (GPT-4o-mini) designed to assist users with business inquiries and conversation. The bot supports conversation memories, business info storage, and summarizes long chat histories.

## Features
- **Multi-modal input processing:** Process text and image URLs.
- **Conversation Memory:** Utilizing buffer and summary memories.
- **MongoDB storage:** Save and retrieve chat history and business info.
- **Customizable AI Tone:** Choose between Friendly, Formal, and Professional.
- **MarkdownV2 Support:** Properly formatted responses for Telegram.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up MongoDB and update credentials if necessary.

## Usage
1. Update the necessary configurations in the `bot.py` file.
2. Run the bot:
   ```bash
   python bot.py
   ```
3. Interact with the bot on Telegram using commands:
   - `/newchat` to start a new session.
   - `/summary` to get a conversation summary.
   - `/settings` to configure AI tone and business info.

## Project Structure
- `/utils` contains helper functions for input formatting and markdown escaping.
- `bot.py` is the main application integrating Telegram Bot API, LangChain, and MongoDB.
- `requirements.txt` lists the necessary Python packages.
- Other configuration files include pre-commit settings and gitignore.

## License
Specify your license information here.

// ...existing notes or additional information...
