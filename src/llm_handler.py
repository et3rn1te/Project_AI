"""
LLM Handler Module

This module handles communication with Ollama (local LLM).

Simple Explanation:
This is your "AI conversation manager" that:
1. Sends prompts to Ollama
2. Gets responses back
3. Handles errors gracefully

Think of it as the phone line to your AI assistant!
"""

import requests
import json
from typing import Optional, Dict
import config


class LLMHandler:
    """
    Handles communication with Ollama LLM.
    
    Simple Explanation:
    This talks to Ollama (your local AI) and gets answers!
    """
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Initialize the LLM handler.
        
        Args:
            model: Ollama model name (e.g., "mistral:7b")
            base_url: Ollama API URL
            temperature: Creativity level (0=focused, 1=creative)
            max_tokens: Maximum response length
            
        Simple Explanation:
        - model: Which AI brain to use
        - temperature: How creative vs focused (0.7 is balanced)
        - max_tokens: How long the answer can be
        """
        self.model = model or config.OLLAMA_MODEL
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.temperature = temperature or config.TEMPERATURE
        self.max_tokens = max_tokens or config.MAX_TOKENS
        
        print(f"🤖 LLM Handler initialized")
        print(f"   Model: {self.model}")
        print(f"   URL: {self.base_url}")
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """
        Generate a response from the LLM.
        
        Simple Explanation:
        Sends your prompt to Ollama and gets an answer back!
        
        Args:
            prompt: The complete prompt to send
            stream: Whether to stream response (not used for now)
            
        Returns:
            The LLM's response text
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        try:
            print(f"🤖 Generating response with {self.model}...")
            
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            answer = result.get('response', '')
            
            print(f"✅ Response generated ({len(answer)} characters)")
            
            return answer
            
        except requests.exceptions.ConnectionError:
            error_msg = (
                "❌ Cannot connect to Ollama!\n"
                "   Make sure Ollama is running.\n"
                "   Try: 'ollama serve' or restart Ollama app"
            )
            print(error_msg)
            return error_msg
            
        except requests.exceptions.Timeout:
            error_msg = "❌ Request timed out. The model might be too slow or stuck."
            print(error_msg)
            return error_msg
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            print(error_msg)
            return error_msg
    
    def check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Simple Explanation:
        Tests if we can talk to Ollama - like checking if the phone line works!
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            print("✅ Ollama is running and accessible")
            return True
            
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {str(e)}")
            return False
    
    def list_models(self) -> list:
        """
        List available Ollama models.
        
        Returns:
            List of model names
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            
            return models
            
        except Exception as e:
            print(f"❌ Error listing models: {str(e)}")
            return []


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 LLM HANDLER TEST")
    print("=" * 60)
    print()
    
    # Initialize handler
    handler = LLMHandler()
    
    # Test connection
    print("\n" + "=" * 60)
    print("TEST 1: Check Connection")
    print("=" * 60)
    
    if handler.check_connection():
        # List available models
        print("\n" + "=" * 60)
        print("TEST 2: List Models")
        print("=" * 60)
        
        models = handler.list_models()
        if models:
            print(f"\nAvailable models:")
            for model in models:
                print(f"  - {model}")
        
        # Test generation
        print("\n" + "=" * 60)
        print("TEST 3: Generate Response")
        print("=" * 60)
        
        test_prompt = """You are a helpful AI assistant.

Question: What is machine learning in one sentence?

Answer:"""
        
        print(f"\nPrompt: {test_prompt}")
        print("\n" + "-" * 60)
        print("Response:")
        print("-" * 60)
        
        response = handler.generate(test_prompt)
        print(response)
        
        print("\n" + "=" * 60)
        print("✅ LLM Handler test complete!")
        print("=" * 60)
    else:
        print("\n⚠️  Ollama is not running. Start it to test LLM generation!")
