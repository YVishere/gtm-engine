"""Main orchestrator for the LLM webscraper."""

import logging
import uuid
from datetime import datetime
from typing import List

from reddit_scraper import RedditScraper
from stackoverflow_scraper import StackOverflowScraper
from llm_processor import LLMProcessor
from models import ScrapingResult, SourceType

class AuthContentScraper:
    """Main orchestrator for scraping and processing auth-related content."""

    def __init__(self):
        self.logger = self._setup_logging()
        self.scrapers = [
            RedditScraper(),
            StackOverflowScraper()
        ]
        self.llm_processor = LLMProcessor()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def run_scraping_session(self) -> ScrapingResult:
        """Run a complete scraping and processing session."""
        session_id = str(uuid.uuid4())[:8]
        self.logger.info(f"Starting scraping session {session_id}")

        # Scrape content from all sources
        all_scraped_content = []
        sources_scraped = []

        for scraper in self.scrapers:
            try:
                with scraper:
                    self.logger.info(f"Scraping {scraper.get_source_type().value}")
                    content = scraper.scrape_recent_content()
                    all_scraped_content.extend(content)
                    sources_scraped.append(scraper.get_source_type())
                    self.logger.info(f"Found {len(content)} items from {scraper.get_source_type().value}")
            except Exception as e:
                self.logger.error(f"Error with scraper {scraper.get_source_type()}: {e}")

        self.logger.info(f"Total scraped items: {len(all_scraped_content)}")

        # Process content with LLM
        self.logger.info("Processing content with LLM")
        processed_content = self.llm_processor.process_content_batch(all_scraped_content)
        self.logger.info(f"Processed {len(processed_content)} items")

        # Generate overall summary
        self.logger.info("Generating overall summary")
        summary_data = self.llm_processor.generate_overall_summary(processed_content)

        # Create final result
        result = ScrapingResult(
            session_id=session_id,
            timestamp=datetime.now(),
            total_items=len(all_scraped_content),
            sources_scraped=sources_scraped,
            processed_content=processed_content,
            overall_summary=summary_data.get('overall_summary', ''),
            top_trends=summary_data.get('top_trends', [])
        )

        self.logger.info(f"Session {session_id} completed successfully")
        return result

    def print_results(self, result: ScrapingResult):
        """Print formatted results to console."""
        print("=" * 80)
        print(f"AUTH CONTENT SCRAPING RESULTS - Session {result.session_id}")
        print("=" * 80)
        print(f"Timestamp: {result.timestamp}")
        print(f"Total Items Found: {result.total_items}")
        print(f"Sources: {', '.join([s.value for s in result.sources_scraped])}")
        print(f"Processed Items: {len(result.processed_content)}")
        print()

        print("OVERALL SUMMARY:")
        print("-" * 40)
        print(result.overall_summary)
        print()

        print("TOP TRENDS:")
        print("-" * 40)
        for i, trend in enumerate(result.top_trends, 1):
            print(f"{i}. {trend}")
        print()

        print("DETAILED FINDINGS:")
        print("-" * 40)
        for i, content in enumerate(result.processed_content[:5], 1):  # Show top 5
            print(f"{i}. {content.original.title}")
            print(f"   Source: {content.original.source.value}")
            print(f"   Relevance: {content.relevance_score:.2f}")
            print(f"   Urgency: {content.urgency_level}")
            print(f"   Summary: {content.summary[:150]}...")
            print(f"   URL: {content.original.url}")
            print()

def main():
    """Main entry point."""
    scraper = AuthContentScraper()

    try:
        result = scraper.run_scraping_session()
        scraper.print_results(result)

    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()
