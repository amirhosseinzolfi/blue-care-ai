# Blue-Care AI Telegram Bot

Blue-Care AI is a Telegram-based chatbot that leverages advanced AI integrations using LangChain and other modern tools to provide personalized, supportive responses for parents. The bot processes text queries, maintains conversational memory, and dynamically summarizes long discussions — all while logging and tracking interactions in a user-friendly manner.

## Features

- **Telegram Interaction:** Built with Telebot, seamlessly interacts with users over Telegram.
- **Advanced AI Integration:** Uses LangChain-powered models (ChatOpenAI and OllamaEmbeddings) to generate context-aware responses.
- **Custom Tools:** Includes example tools (e.g., weather info) that showcase extensible functionality.
- **User Memory Management:** Stores conversational history and memories in JSON files under `user_data` to persist context.
- **Conversation Summarization:** Automatically summarizes long conversations for improved performance.
- **Rich Logging:** Utilizes the Rich library to display attractive, structured logs in the console.
- **Customizable Settings:** Allows users to customize settings like AI tone and input personal kid information via commands.

## Project Structure

```
/blue-care-ai
│   bot2.py           # Main Telegram bot entry point
│   ai_prompts.py     # AI prompt templates (extendable as needed)
│   requirements.txt  # List of Python dependencies
│   README.md         # This comprehensive project overview
│   .gitignore        # Git ignore settings
│
└───user_data         # Persists user memories and chat history (ignored by Git)
```

## Installation

1. **Clone the repository:**

   ```
   git clone <your-repo-url>
   cd /d:/Documents/programming projects/telegram bots/blue-care-ai
   ```

2. **Create a virtual environment (optional but recommended):**

   ```
   python -m venv venv
   source venv/bin/activate    # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**

   Set up your environment variables for:
   
   - `TELEGRAM_BOT_TOKEN`: Your Telegram bot token.
   - `OPENAI_API_KEY`: Your OpenAI API key.

   You may set these globally or create an `.env` file (ensure it is added to `.gitignore`).

## Usage

1. **Run the Bot:**

   ```
   python bot2.py
   ```

2. **Commands Overview:**
   - `/start`: Initiates the bot and provides a welcome message.
   - `/help`: Displays instructions and usage tips.
   - `/setting`: Allows you to update your personal details such as kid information and the desired AI response tone.

3. **Internal Processes:**
   - The bot logs user interactions, processes messages via an AI agent, and maintains a memory history.
   - In longer conversations, it automatically summarizes interactions to optimize performance.

## Contributing

Contributions and suggestions are welcome!  
Feel free to fork the repository, create a feature branch, and submit a pull request. Please ensure your code adheres to the project's style and include useful comments.

## License

Specify your license information here.

## Acknowledgements

- Built with Telebot, Rich, LangChain, and other open-source libraries.
- Thanks to all contributors and the open-source community for their valuable tools and feedback.

Happy Coding!
