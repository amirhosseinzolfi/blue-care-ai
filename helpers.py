
def escape_markdown_v2(text: str) -> str:
    """
    Escape special characters for Telegram's MarkdownV2 format.
    """
    # Characters that need to be escaped in MarkdownV2
    special_chars = ['_', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    escaped_text = text
    for char in special_chars:
        escaped_text = escaped_text.replace(char, f"\\{char}")
    
    return escaped_text

def refine_ai_response(response: str) -> str:
    """
    Clean and format the AI response for better presentation.
    """
    # Remove any unnecessary whitespace or newlines
    cleaned = response.strip()
    
    # Handle empty responses
    if not cleaned:
        return "I apologize, but I couldn't generate a proper response."
    
    return cleaned
