import os
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Define request model
class QueryRequest(BaseModel):
    query: str

def truncate_to_token_limit(text: str, max_tokens: int = 6000, buffer: int = 500) -> str:
    """
    Truncate the input text to fit within token limit.
    Approximate 1 token ≈ 4 characters.
    The buffer reserves tokens for prompt/query content.
    """
    max_chars = (max_tokens - buffer) * 4
    return text[:max_chars]

async def generate_response_from_groq(input_text: str, query: str = "", custom_prompt: str = None) -> str:
    base_prompt = custom_prompt or (
        "You are an intelligent assistant. Analyze the following text extracted from technical or business documents.\n\n"
        "Your task is to generate a clear, concise, and well-structured summary that:\n"
        "- Extracts only the most relevant and informative content\n"
        "- Highlights important facts, figures, entities, and topics\n"
        "- Groups related information logically using bullet points or short paragraphs\n"
        "- Avoids any introductory or concluding phrases like 'Here is a summary'\n"
        "- Excludes image URLs, file paths, or irrelevant metadata\n"
        "- Maintains a professional and factual tone, without speculation\n\n"
        "Always answer in well-formatted text that is easy to read and understand.\n\n"
    )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")

    # Truncate input to avoid exceeding token limits
    truncated_input = truncate_to_token_limit(input_text)

    full_input = f"{base_prompt}User Query: {query}\n\nJSON Data:\n{truncated_input}"

    llm = ChatGroq(
        model_name="llama3-70b-8192",
        api_key=groq_api_key
    )

    messages = [HumanMessage(content=full_input)]
    response = llm.invoke(messages)

    return response.content
