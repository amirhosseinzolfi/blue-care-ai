# ...existing code (if any)...

from langchain_openai import ChatOpenAI
# ...import other modules needed for the prompt...

llm_for_kid_info = ChatOpenAI(
    base_url="http://localhost:15203/v1",
    model_name="gemini-1.5-flash",
    temperature=0.3
)

def analyze_and_structure_kid_info(new_info: str, old_info: str = "") -> str:
    """
    Combine old_info (if any) and new_info, then ask the LLM to clean, structure,
    and format the result in Persian, returning a neatly organized string.
    """
    combined = old_info + "\n" + new_info if old_info else new_info
    prompt = (
        "You are a Persian assistant. Take the user's child info below, then clean,"
        " organize, and structure it in an attractive, well-formatted Persian text."
        "\n\nChild info:\n" + combined
    )
    response = llm_for_kid_info.invoke(prompt)
    return response.content.strip()
