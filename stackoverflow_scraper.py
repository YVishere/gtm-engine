"""StackOverflow scraper implementation."""

import json
from datetime import datetime
from typing import List
import time
from urllib.parse import urlencode

from base_scraper import BaseScraper
from models import ScrapedContent, SourceType

class StackOverflowScraper(BaseScraper):
    """Scraper for StackOverflow authentication-related questions."""

    def __init__(self):
        super().__init__()
        self.base_url = "https://api.stackexchange.com/2.3"
        self.site = "stackoverflow"
        
        # Log API key status
        if self.config.STACKOVERFLOW_API_KEY:
            self.logger.info("StackOverflow API key loaded - enhanced rate limits available")
        else:
            self.logger.warning("No StackOverflow API key - using default rate limits")

    def get_source_type(self) -> SourceType:
        """Return StackOverflow as source type."""
        return SourceType.STACKOVERFLOW

    def scrape_recent_content(self) -> List[ScrapedContent]:
        """Scrape recent StackOverflow questions about authentication."""
        all_content = []

        # Get recent questions with auth-related tags
        for tag in self.config.STACKOVERFLOW_TAGS:
            self.logger.info(f"Scraping StackOverflow for tag: {tag}")
            tag_content = self._scrape_tag_questions(tag)
            all_content.extend(tag_content)

            # Rate limiting - reduced since we're using API key
            delay = 1 if self.config.STACKOVERFLOW_API_KEY else 2
            time.sleep(delay)

        return all_content

    def _scrape_tag_questions(self, tag: str) -> List[ScrapedContent]:
        """Scrape questions for a specific tag."""
        url = f"{self.base_url}/questions"

        params = {
            'site': self.site,
            'tagged': tag,
            'sort': 'creation',
            'order': 'desc',
            'pagesize': min(self.config.MAX_RESULTS_PER_SOURCE, 100),
            'filter': 'withbody'  # Include question body
        }
        
        # Add API key if available for higher rate limits
        if self.config.STACKOVERFLOW_API_KEY:
            params['key'] = self.config.STACKOVERFLOW_API_KEY
            self.logger.debug(f"Using StackOverflow API key for enhanced rate limits")
        else:
            self.logger.warning("No StackOverflow API key found - using limited rate limits")

        response = self._make_request(url, params=params)
        if not response:
            return []

        try:
            data = response.json()
            questions = data.get('items', [])

            content_list = []
            for question in questions:
                content = self._parse_question(question)

                if content and self._is_recent(content.timestamp):
                    if self._is_auth_related(f"{content.title} {content.content}"):
                        content_list.append(content)

            return content_list

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Error parsing StackOverflow data for tag {tag}: {e}")
            return []

    def _parse_question(self, question: dict) -> ScrapedContent:
        """Parse a StackOverflow question into ScrapedContent."""
        timestamp = datetime.fromtimestamp(question.get('creation_date', 0))

        return ScrapedContent(
            title=question.get('title', ''),
            content=question.get('body', ''),
            url=question.get('link', ''),
            source=SourceType.STACKOVERFLOW,
            timestamp=timestamp,
            author=question.get('owner', {}).get('display_name', 'unknown'),
            score=question.get('score', 0),
            comments_count=question.get('answer_count', 0),
            tags=question.get('tags', [])
        )
