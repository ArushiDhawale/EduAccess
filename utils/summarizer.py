import os
from google import genai

def generate_lecture_summary(text_content: str, api_key: str) -> str:
    """Generates structured notes, key takeaways, and a concise summary."""
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an AI learning assistant for inclusive classroom education.
    Analyze the following lecture transcript or study material and generate:
    
    1. 📌 **Executive Summary** (2-3 concise paragraphs)
    2. 🔑 **Key Takeaways & Concepts** (Bullet points)
    3. 📖 **Important Terminology & Simple Definitions**
    4. ❓ **Quick Revision Questions** (3-4 self-test questions)

    Content:
    \"\"\"{text_content}\"\"\"
    """
    
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text

def simplify_content(text_content: str, api_key: str) -> str:
    """Simplifies complex text and concepts using simple, accessible language."""
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an AI learning assistant. Your task is to simplify the following educational content or concepts.
    - Rewrite complex sentences into shorter, simpler ones.
    - Explain any technical terminology or jargon in simple terms.
    - Use formatting like bullet points, clear headings, and analogies to make it easy to follow.
    - Keep the original meaning fully intact but make it accessible for diverse learning needs.
    
    Content to simplify:
    \"\"\"{text_content}\"\"\"
    """
    
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text