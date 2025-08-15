"""Configuration settings for the LLM webscraper."""

from datetime import datetime, timedelta
from typing import List

class Config:
    """Configuration class for scraper settings."""

    # LLM Settings
    OLLAMA_MODEL = "llama3.2:1b"
    OLLAMA_HOST = "http://localhost:11434"

    # Scraping Settings
    TIME_WINDOW_MINUTES = 5
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
