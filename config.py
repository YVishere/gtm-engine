"""Configuration settings for the LLM webscraper."""

import os
from datetime import datetime, timedelta
from typing import List

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class Config:
    """Configuration class for scraper settings."""

    # LLM Settings
    OLLAMA_MODEL = "llama3.2:1b"  # Change to "deepseek-r1:1.5b" for DeepSeek R1
    OLLAMA_HOST = "http://localhost:11434"
    
    # Model-specific settings
    MODEL_CONFIGS = {
        "llama3.2:1b": {
            "timeout": 30,
            "format": "json"
        },
        "deepseek-r1:1.5b": {
            "timeout": 60,  # DeepSeek R1 may need more time for thinking
            "format": None  # Don't force JSON format, handle in parsing
        },
        "llama3.1:8b": {
            "timeout": 30,
            "format": "json"
        }
    }

    # API Keys from environment
    STACKOVERFLOW_API_KEY = os.getenv('STACK_OVERFLOW_API')
    GITHUB_API_KEY = os.getenv('GITHUB_API')

    # Scraping Settings
    TIME_WINDOW_MINUTES = 60 * 24 * 30 * 5  # 5 months for testing to get more data
    MAX_RESULTS_PER_SOURCE = 50
    REQUEST_TIMEOUT = 10

    # Keywords for authentication-related content
    AUTH_KEYWORDS = [
        'authentication', 'auth', 'login', 'session', 'jwt', 'oauth',
        'security', 'password', 'token', 'sso', 'saml', 'openid',
        'authorize', 'credential', 'signin', 'signup'
    ]

    # Reddit Settings
    REDDIT_USER_AGENT = "AuthScraper/1.0"
    REDDIT_SUBREDDITS = [
        'programming', 'webdev', 'javascript', 'python', 'security',
        'sysadmin', 'devops', 'node', 'reactjs'
    ]

    # StackOverflow Settings
    STACKOVERFLOW_TAGS = [
        'authentication', 'oauth', 'jwt', 'session', 'security',
        'login', 'authorization', 'saml', 'openid'
    ]

    @classmethod
    def get_time_threshold(cls) -> datetime:
        """Get the datetime threshold for recent content."""
        return datetime.now() - timedelta(minutes=cls.TIME_WINDOW_MINUTES)
