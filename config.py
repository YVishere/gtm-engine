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
    OLLAMA_MODEL = "llama3.1:8b"  # Change to "deepseek-r1:1.5b" for DeepSeek R1
    OLLAMA_HOST = "http://localhost:11434"
    
    # Model-specific settings
    MODEL_CONFIGS = {
        "llama3.2:3b": {
            "timeout": 30,
            "format": "json"
        },
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

    # GitHub API Rate Limiting Settings
    GITHUB_MAX_REQUESTS_PER_SESSION = 15  # Conservative limit for search operations
    GITHUB_MAX_REQUESTS_PER_HOUR = 100    # GitHub allows more, but we stay conservative
    GITHUB_REQUEST_DELAY = 1              # Seconds between requests
    
    # LLM Decision Making Settings
    LLM_MAX_RETRIES = 2
    LLM_TIMEOUT_SECONDS = 30
    LLM_CONFIDENCE_THRESHOLD = 0.7        # Minimum confidence for LLM decisions
    
    # RAG Search Strategy Settings
    MAX_SEARCH_QUERIES_PER_OPPORTUNITY = 4
    MAX_REPOSITORIES_TO_ANALYZE = 5
    REPOSITORY_ANALYSIS_DEPTH_DEFAULT = "medium"  # shallow, medium, deep
    
    # Scraping Settings
    TIME_WINDOW_MINUTES = 60 * 24 * 30 * 5  # 5 months for testing to get more data
    MAX_RESULTS_PER_SOURCE = 200
    REQUEST_TIMEOUT = 10

    # Vector Processing Settings
    USE_VECTORIZATION = True
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, lightweight model
    MIN_CLUSTER_SIZE = 3
    CLUSTER_SIMILARITY_THRESHOLD = 0.75
    REPRESENTATIVE_SAMPLE_SIZE = 1  # Per cluster
    BATCH_EMBEDDING_SIZE = 100
    MAX_CLUSTERS_TO_PROCESS = 100
    UMAP_COMPONENTS = 50  # Dimensionality reduction

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
