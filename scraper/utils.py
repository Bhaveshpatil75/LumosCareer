import os
import requests
from pdfminer.high_level import extract_text
from django.core.exceptions import ValidationError

def validate_file_size(file):
    """
    Validates that the file size is less than 5MB.
    """
    max_size_mb = 5
    if file.size > max_size_mb * 1024 * 1024:
        raise ValidationError(f"File size must be under {max_size_mb}MB")

import io

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
