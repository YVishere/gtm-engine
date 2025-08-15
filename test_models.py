#!/usr/bin/env python3
"""Test script for different Ollama models."""

import logging
from models import ScrapedContent, SourceType
from llm_processor import LLMProcessor
from config import Config
from datetime import datetime

def test_model_parsing():
    """Test LLM processing with sample content."""
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create sample content
    sample_content = ScrapedContent(
        title="JWT token refresh best practices",
        content="I'm building a React app and struggling with JWT token refresh. Should I store the refresh token in localStorage or httpOnly cookies? What's the most secure approach for handling token expiration?",
        url="https://stackoverflow.com/test",
        source=SourceType.STACKOVERFLOW,
        timestamp=datetime.now(),
        author="testuser",
        score=5,
        comments_count=3,
        tags=["jwt", "authentication"]
    )
    
    # Test current model
    processor = LLMProcessor()
    print(f"Testing model: {Config.OLLAMA_MODEL}")
    print("-" * 50)
    
    try:
        result = processor._process_single_content(sample_content)
        if result:
            print("✅ Processing successful!")
            print(f"Summary: {result.summary}")
            print(f"Relevance: {result.relevance_score}")
            print(f"Topics: {result.key_topics}")
            print(f"Urgency: {result.urgency_level}")
        else:
            print("❌ Processing failed")
    except Exception as e:
        print(f"❌ Error: {e}")

def print_model_switch_instructions():
    """Print instructions for switching models."""
    print("\n" + "="*60)
    print("MODEL SWITCHING INSTRUCTIONS")
    print("="*60)
    print("\n1. For DeepSeek R1, first pull the model:")
    print("   ollama pull deepseek-r1:1.5b")
    print("\n2. Edit config.py and change:")
    print('   OLLAMA_MODEL = "deepseek-r1:1.5b"')
    print("\n3. The system will automatically:")
    print("   - Handle DeepSeek's thinking output format")
    print("   - Use longer timeouts for processing")
    print("   - Parse JSON from various response formats")
    print("\n4. Run the test again to verify:")
    print("   python test_models.py")

if __name__ == "__main__":
    test_model_parsing()
    print_model_switch_instructions()
