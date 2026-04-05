import os
import requests
from pdfminer.high_level import extract_text
from django.core.exceptions import ValidationError
import io

def validate_file_size(file):
    """
    Validates that the file size is less than 5MB.
    """
    max_size_mb = 5
    if file.size > max_size_mb * 1024 * 1024:
        raise ValidationError(f"File size must be under {max_size_mb}MB")

def extract_text_from_pdf_file(uploaded_file):
    """
    Extracts text from an uploaded PDF file in memory or temporary storage.
    Pdfminer likes paths, but for InMemoryUploadedFile we might need to handle it.
    """
    try:
        # Reset file pointer to the beginning
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
            
        # Read content into a BytesIO stream which `extract_text` is known to accept
        file_stream = io.BytesIO(uploaded_file.read())
        text = extract_text(file_stream)
        
        # Reset file pointer for subsequent operations (like saving the model)
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
            
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def send_to_n8n_webhook(url, payload):
    """
    Sends a payload to an n8n webhook.
    """
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error sending to n8n: {e}")
        return None

def call_gemini_api(messages, system_instruction=None, max_tokens=4096):
    """
    Calls Google Gemini API.
    messages: list of dicts [{'role': 'user'|'model', 'content': 'text'}]
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}"
    
    contents = []
    for msg in messages:
        # Map roles: 'user' -> 'user', 'ai'/'model' -> 'model'
        role = "user" if msg.get('role') == 'user' else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg.get('content', '')}]
        })
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": max_tokens,
        }
    }

    if system_instruction:
        payload["systemInstruction"] = {
             "parts": [{"text": system_instruction}]
        }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        # Extract text from the first candidate
        parts = data.get('candidates', [{}])[0].get('content', {}).get('parts', [])
        if parts:
            return parts[0].get('text', '')
        return "I'm not sure how to respond to that."
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return None


def call_grok_api(messages, system_instruction=None, max_tokens=4096):
    """
    Calls Groq API (OpenAI-compatible format) for blazing fast LPU inference.
    Primary chat LLM — faster and more efficient than Gemini.
    Falls back to None if unavailable so caller can try Gemini.
    """
    api_key = os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found, skipping Groq.")
        return None

    # Using Groq's endpoint instead of xAI
    url = "https://api.groq.com/openai/v1/chat/completions"

    oai_messages = []
    if system_instruction:
        oai_messages.append({"role": "system", "content": system_instruction})

    for msg in messages:
        role = "user" if msg.get('role') == 'user' else "assistant"
        oai_messages.append({"role": role, "content": msg.get('content', '')})

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": oai_messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        choices = data.get('choices', [])
        if choices:
            return choices[0].get('message', {}).get('content', '')
        return None
    except Exception as e:
        print(f"Error calling Groq: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response: {response.text}")
        return None


def call_chat_api(messages, system_instruction=None, max_tokens=4096):
    """
    Primary chat function: tries Gemini first, falls back to Groq.
    Used for interview chat and therapy chat.
    """
    # Try Gemini first
    reply = call_gemini_api(messages, system_instruction, max_tokens)
    if reply:
        return reply

    # Fallback to Groq
    print("Gemini unavailable, falling back to Groq.")
    return call_grok_api(messages, system_instruction, max_tokens)


def call_gemini_with_rag(messages, context_query, system_instruction_base=None, max_tokens=4096):
    """
    Enhanced Gemini call that retrieves RAG context before generating.
    1. Retrieves relevant knowledge via RAG engine
    2. Injects context into system instruction
    3. Calls Gemini with enriched prompt
    """
    try:
        from .rag_engine import get_rag_engine
        rag = get_rag_engine()
        rag_context = rag.build_context(context_query, max_chars=2500)
    except Exception as e:
        print(f"RAG retrieval failed (non-fatal): {e}")
        rag_context = ""

    if system_instruction_base and rag_context:
        enhanced_instruction = f"{system_instruction_base}\n\n{rag_context}"
    elif rag_context:
        enhanced_instruction = rag_context
    else:
        enhanced_instruction = system_instruction_base

    return call_gemini_api(messages, enhanced_instruction, max_tokens=max_tokens)


def call_mcp_tool(tool_name, **kwargs):
    """
    Invokes an MCP tool by name and returns structured results.
    """
    try:
        from .mcp_server import get_mcp_server
        server = get_mcp_server()
        return server.call_tool(tool_name, kwargs)
    except Exception as e:
        print(f"MCP tool call failed: {e}")
        return {"error": str(e)}
