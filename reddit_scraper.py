"""Reddit scraper implementation."""

import json
from datetime import datetime
from typing import List
import time

from base_scraper import BaseScraper
from models import ScrapedContent, SourceType

class RedditScraper(BaseScraper):
    """Scraper for Reddit authentication-related posts."""

    def __init__(self):
        super().__init__()
        self.base_url = "https://www.reddit.com"
        self.headers = {
            'User-Agent': self.config.REDDIT_USER_AGENT
        }

    def get_source_type(self) -> SourceType:
        """Return Reddit as source type."""
        return SourceType.REDDIT

    def scrape_recent_content(self) -> List[ScrapedContent]:
        """Scrape recent Reddit posts about authentication."""
        all_content = []

        for subreddit in self.config.REDDIT_SUBREDDITS:
            self.logger.info(f"Scraping r/{subreddit}")
            subreddit_content = self._scrape_subreddit(subreddit)
            all_content.extend(subreddit_content)

            # Rate limiting
            time.sleep(1)

        return all_content

    def _scrape_subreddit(self, subreddit: str) -> List[ScrapedContent]:
        """Scrape a specific subreddit."""
        url = f"{self.base_url}/r/{subreddit}/new.json"
        params = {'limit': self.config.MAX_RESULTS_PER_SOURCE}

        response = self._make_request(url, headers=self.headers, params=params)
        if not response:
            return []

        try:
            data = response.json()
            posts = data.get('data', {}).get('children', [])

            content_list = []
            for post_data in posts:
                post = post_data.get('data', {})
                content = self._parse_post(post, subreddit)

                if content and self._is_auth_related(f"{content.title} {content.content}"):
                    if self._is_recent(content.timestamp):
                        content_list.append(content)

            return content_list

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Error parsing Reddit data for r/{subreddit}: {e}")
            return []

    def _parse_post(self, post: dict, subreddit: str) -> ScrapedContent:
        """Parse a Reddit post into ScrapedContent."""
        import html
        
        timestamp = datetime.fromtimestamp(post.get('created_utc', 0))
        
        # Clean HTML entities from title and content
        title = html.unescape(post.get('title', ''))
        content = html.unescape(post.get('selftext', ''))

        return ScrapedContent(
            title=title,
            content=content,
            url=f"{self.base_url}{post.get('permalink', '')}",
            source=SourceType.REDDIT,
            timestamp=timestamp,
            author=post.get('author', 'unknown'),
            score=post.get('score', 0),
            comments_count=post.get('num_comments', 0),
            tags=[f"r/{subreddit}"]
        )
