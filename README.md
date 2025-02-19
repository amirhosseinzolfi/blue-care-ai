# Blue-Care AI Telegram Bot

Welcome to **Blue-Care AI** – your cutting-edge Telegram assistant dedicated to supporting parents! Designed with advanced AI technology and a friendly interface, Blue-Care AI not only listens but also provides personalized, practical advice to help you nurture your child with confidence.

## Why Blue-Care AI?
- **Innovative AI Integration:** Utilizes LangChain, OpenAI, and Ollama to deliver smart, context-aware responses.
- **User-Centric Experience:** Remembers conversations and tailors suggestions based on your unique child information.
- **Effortless Interaction:** Enjoy seamless, real-time communication via Telegram.
- **Optimized Performance:** Automatically summarizes long conversations to keep interactions concise.
- **Stunning Console Visuals:** Benefit from attractive logging and structured outputs using the Rich library.

## Features
- **Telegram Bot Interaction:** Built with Telebot for responsive communication.
- **AI-Powered Responses:** Provides warm, precise answers using advanced AI tools.
- **Memory & History Management:** Keeps track of chats and child details securely in JSON files.
- **Dynamic Conversation Summaries:** Automatically generates summaries to enhance performance.
- **Customizable Settings:** Tailor the bot's behavior to your family's needs.

## Project Structure
```
/blue-care-ai
│   bot2.py           // Main Telegram bot entry point
│   ai_prompts.py     // Customizable AI prompt templates
│   requirements.txt  // Python dependencies list
│   README.md         // Project overview (this file)
│   .gitignore        // Git ignore settings
│
└───user_data         // User-specific data: chat history and memories
```

## Live Demo
Experience Blue-Care AI in action by [visiting our demo link](#) *(coming soon)* or deploy locally by following the steps below!

## Installation
1. **Clone the Repository:**
   ```
   git clone <your-repo-url>
   cd "/d:/Documents/programming projects/telegram bots/blue-care-ai"
   ```
2. **Set Up a Virtual Environment:**
   - On Unix:
     ```
     python -m venv venv
     source venv/bin/activate
     ```
   - On Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
3. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```
4. **Configure Environment Variables:**
   - Ensure you have `TELEGRAM_BOT_TOKEN` and `OPENAI_API_KEY` set via your system or an `.env` file.

## Usage
- **Start the Bot:**
  ```
  python bot2.py
  ```
- **Key Commands:**
  - `/start` — Begin your journey with Blue-Care AI.
  - `/help` — Get detailed usage instructions.
  - `/setting` — Customize your preferences (e.g., child info, AI tone).

## Community & Support
Join our growing community of modern parents and tech enthusiasts! Share your feedback, report issues, or contribute improvements. We’re here to support you every step of the way.

## Contributing
o Contributions are welcome! Please fork the repository, create a feature branch, and submit your pull requests. Follow coding standards and document your changes.

## License
Specify your license information here.

## Acknowledgements
- Proudly built with Telebot, LangChain, Ollama, and Rich.
- Special thanks to the open-source community and contributors who make innovative digital parenting a reality.

Happy Coding and Parenting!
