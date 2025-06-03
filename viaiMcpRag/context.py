import os
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import re
from llm_provider import get_llm_provider

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Define request model
class QueryRequest(BaseModel):
    query: str
    llm_provider: str = "groq"  # Default to groq for backward compatibility

def truncate_to_token_limit(text: str, max_tokens: int = 30000, buffer: int = 500) -> str:
    """
    Truncate the input text to fit within token limit.
    Approximate 1 token ≈ 4 characters.
    The buffer reserves tokens for prompt/query content.
    """
    max_chars = (max_tokens - buffer) * 4
    return text[:max_chars]

import re
import logging

log = logging.getLogger(__name__)

def is_greeting(query: str) -> bool:
    """Detects if the user query is a greeting."""
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    return query.lower().strip() in greetings

async def generate_response_from_llm(
    input_text: str,
    query: str = "",
    custom_prompt: str = None,
    llm_provider: str = "groq"
) -> str:
    if is_greeting(query):
        return "Hello! How can I assist you today."

    # Dynamic prompt based on provider
    if custom_prompt:
        base_prompt = custom_prompt
    elif llm_provider.lower() == "openai":
        base_prompt = (
            "You are an expert query solver. Provide a concise response using only the information in the data. "
            "Use the JSON data when available,"
            "If the query is not addressed, say that the information is not found in the data."
        )
    else:
        base_prompt = (
            "You are an expert query solver. Provide a concise response using only the information in the data. "
            "Do not use general knowledge or define topics unless explicitly mentioned. "
            "If the query is not addressed, say that the information is not found in the data."
        )

    provider = get_llm_provider(llm_provider)
    log.info(f"Using {llm_provider} provider for response generation")

    truncated_input = truncate_to_token_limit(input_text)
    full_input = f"{base_prompt}\n\nUser Query: {query}\n\nJSON Data:\n{truncated_input}"

    response = await provider.generate_response(full_input)
    cleaned_response = re.sub(r'\s+', ' ', response).strip()

    fallback_patterns = [
        r"\b(not mentioned|not found|does not appear|no relevant data|cannot find)\b",
        r"(?i)this appears to be",
        r"(?i)data.*seems.*about",
        r"(?i)i cannot find"
    ]

    if any(re.search(pattern, cleaned_response) for pattern in fallback_patterns):
        log.warning("Fallback detected, generating concise document title.")

        # Prompt to request only a short title
        fallback_prompt = (
            "Return only a short, clear title (2 to 5 words) that summarizes the overall content of the document. "
            "Do not include explanations or extra context. Only return the title."
        )
        fallback_input = f"{fallback_prompt}\n\nJSON Data:\n{truncated_input}"
        fallback_summary = await provider.generate_response(fallback_input)
        fallback_summary = re.sub(r'\s+', ' ', fallback_summary).strip().strip('"').strip('.')

        return (
            "no data found for your query."
            # + fallback_summary
        )

    return cleaned_response


async def summarize_extracted_text(input_text: str, custom_prompt: str = None, llm_provider: str = "groq") -> str:
    summarization_prompt = custom_prompt or (
        "You are a highly skilled AI tasked with generating a comprehensive, structured summary of the following input text.\n\n"
        "Your goal is to extract and present all significant topics, subtopics, arguments, facts, and data points in detail. Do not omit key elements, even if the text is long.\n\n"
        "Instructions:\n"
        "- Cover every major topic and subtopic thoroughly, explaining the context and key points under each.\n"
        "- Preserve logical flow and structure, using bullet points or sections if appropriate.\n"
        "- Highlight factual data, technical information, definitions, and examples clearly.\n"
        "- Avoid vague generalizations — be precise and detailed.\n"
        "- Do not rewrite or paraphrase too abstractly; retain specific terminology where relevant.\n"
        "- Maintain objectivity and clarity, without editorializing or simplifying too much.\n"
        "- If there is structured content (tables, code, JSON, lists), describe their purpose and key elements accurately.\n"
        "- Do NOT include phrases like 'In conclusion' or 'The text discusses...'; simply present the extracted information.\n\n"
        "Output a detailed, accurate, and complete summary suitable for someone who needs a full understanding of the original content without reading it."
    )

    # Get the appropriate LLM provider
    provider = get_llm_provider(llm_provider)
    log.info(f"Using {llm_provider} provider for text summarization")

    # Truncate input to fit within token limits
    truncated_input = truncate_to_token_limit(input_text)

    full_input = f"{summarization_prompt}\n\nInput Text:\n{truncated_input}\n\nSummary:"

    # Generate summary using the provider
    return await provider.generate_response(full_input)