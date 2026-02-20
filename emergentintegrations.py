"""
Compatibility module for emergentintegrations
Uses Google Generative AI (gemini-2.5-flash) as backend
"""
import google.generativeai as genai
from google.generativeai import GenerativeModel
import json
import os
from pathlib import Path


# Configure the Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class FileContentWithMimeType:
    """Compatibility class for file content with mime type"""
    def __init__(self, file_path: str, mime_type: str):
        self.file_path = file_path
        self.mime_type = mime_type


class UserMessage:
    """Compatibility class for user message"""
    def __init__(self, text: str = None, file_contents: list = None):
        self.text = text
        self.file_contents = file_contents or []


class LlmChat:
    """Compatibility class for LlmChat using Google Gemini"""
    
    def __init__(self, api_key: str = None, session_id: str = None, system_message: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.session_id = session_id
        self.system_message = system_message
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        self.model = None
    
    def with_model(self, provider: str, model_name: str):
        """Set the model to use"""
        # Map model names
        if model_name == "gemini-2.5-flash":
            model_name = "gemini-2.0-flash"
        
        self.model = GenerativeModel(
            model_name,
            system_instruction=self.system_message
        )
        return self
    
    async def send_message(self, message: UserMessage):
        """Send a message and get response"""
        if not self.model:
            raise ValueError("Model not set. Call with_model() first.")
        
        # Build content
        contents = []
        
        if message.text:
            contents.append(message.text)
        
        # Handle file contents
        for fc in message.file_contents:
            if isinstance(fc, FileContentWithMimeType):
                try:
                    # Upload file to Gemini
                    file_blob = genai.upload_file(fc.file_path, mime_type=fc.mime_type)
                    # Wait for file to be ACTIVE (required by Gemini API)
                    import time
                    max_wait = 60  # Maximum 60 seconds wait
                    wait_time = 0
                    while file_blob.state.name == "PROCESSING" and wait_time < max_wait:
                        time.sleep(2)
                        wait_time += 2
                        file_blob = genai.get_file(file_blob.name)
                    if file_blob.state.name != "ACTIVE":
                        raise ValueError(f"File not active after upload. State: {file_blob.state.name}")
                    contents.append(file_blob)
                except Exception as e:
                    print(f"Error uploading file: {e}")
                    # Fallback to text if file upload fails
                    if Path(fc.file_path).exists():
                        try:
                            contents.append(Path(fc.file_path).read_text()[:8000])
                        except:
                            pass
        
        # Generate response
        response = self.model.generate_content(contents)
        
        return response.text



# For backward compatibility
__all__ = ['LlmChat', 'UserMessage', 'FileContentWithMimeType']
