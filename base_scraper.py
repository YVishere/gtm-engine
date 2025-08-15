"""Abstract base class for web scrapers."""

from abc import ABC, abstractmethod
from typing import List, Optional
import requests
from datetime import datetime
import logging

from models import ScrapedContent, SourceType
from config import Config

class BaseScraper(ABC):
    """Abstract base class for content scrapers."""

    def __init__(self):
        self.config = Config()
        self.session = requests.Session()
        self.session.timeout = self.config.REQUEST_TIMEOUT
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def get_source_type(self) -> SourceType:
        """Return the source type for this scraper."""
        pass

    @abstractmethod
    def scrape_recent_content(self) -> List[ScrapedContent]:
        """Scrape recent authentication-related content."""
        pass

    def _is_auth_related(self, text: str) -> bool:
        """Check if text contains authentication-related keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.config.AUTH_KEYWORDS)

    def _is_recent(self, timestamp: datetime) -> bool:
        """Check if content is within the time window."""
        return timestamp >= self.config.get_time_threshold()

    def _make_request(self, url: str, headers: dict = None, params: dict = None) -> Optional[requests.Response]:
        """Make HTTP request with error handling."""
        try:
            response = self.session.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
