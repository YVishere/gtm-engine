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
        """Run a complete scraping and processing session with progress tracking."""
        session_id = str(uuid.uuid4())[:8]
        self.logger.info(f"Starting scraping session {session_id}")
        
        print("🚀 Starting Authentication Content Intelligence Session")
        print("=" * 60)
        print(f"Session ID: {session_id}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Scrape content from all sources
        all_scraped_content = []
        sources_scraped = []

        print("📡 Data Collection Phase")
        print("-" * 30)
        
        for i, scraper in enumerate(self.scrapers, 1):
            try:
                with scraper:
                    source_name = scraper.get_source_type().value
                    print(f"[{i}/{len(self.scrapers)}] Scraping {source_name}...")
                    
                    content = scraper.scrape_recent_content()
                    all_scraped_content.extend(content)
                    sources_scraped.append(scraper.get_source_type())
                    
                    print(f"   ✅ Found {len(content)} items from {source_name}")
            except Exception as e:
                print(f"   ❌ Error with {scraper.get_source_type().value}: {e}")
                self.logger.error(f"Error with scraper {scraper.get_source_type()}: {e}")

        print(f"\n📊 Collection Summary: {len(all_scraped_content)} total items scraped")
        
        if not all_scraped_content:
            print("⚠️  No content found. Check time window and keywords.")
            return self._create_empty_result(session_id, sources_scraped)

        # Process content with LLM
        print(f"\n🧠 AI Analysis Phase")
        print("-" * 30)
        processed_content = self.llm_processor.process_content_batch(all_scraped_content)

        # Generate overall summary
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

        print(f"\n🎉 Session {session_id} completed successfully!")
        print(f"📈 Performance: {len(all_scraped_content)} scraped → {len(processed_content)} analyzed")
        return result

    def _create_empty_result(self, session_id: str, sources_scraped: List[SourceType]) -> ScrapingResult:
        """Create empty result when no content is found."""
        return ScrapingResult(
            session_id=session_id,
            timestamp=datetime.now(),
            total_items=0,
            sources_scraped=sources_scraped,
            processed_content=[],
            overall_summary="No relevant authentication content found in the specified time window.",
            top_trends=[]
        )

    def print_results(self, result: ScrapingResult):
        """Print formatted results to console with enhanced formatting."""
        print("\n" + "=" * 80)
        print(f"🎯 AUTHENTICATION INTELLIGENCE REPORT - Session {result.session_id}")
        print("=" * 80)
        print(f"📅 Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Data Pipeline: {result.total_items} items → {len(result.processed_content)} analyzed")
        print(f"🔍 Sources: {', '.join([s.value for s in result.sources_scraped])}")
        
        if not result.processed_content:
            print("\n⚠️  No high-relevance content found in this session.")
            return
        
        # Performance metrics
        processing_efficiency = (len(result.processed_content) / result.total_items * 100) if result.total_items > 0 else 0
        print(f"⚡ Processing Efficiency: {processing_efficiency:.1f}% (smart filtering)")
        print()

        print("📋 EXECUTIVE SUMMARY")
        print("-" * 40)
        print(result.overall_summary)
        print()

        print("📈 TRENDING TOPICS")
        print("-" * 40)
        for i, trend in enumerate(result.top_trends[:7], 1):
            print(f"{i:2d}. {trend}")
        print()

        print("🔥 HIGH-VALUE OPPORTUNITIES")
        print("-" * 40)
        
        # Sort by relevance and show top items
        top_items = sorted(result.processed_content, key=lambda x: x.relevance_score, reverse=True)[:8]
        
        for i, content in enumerate(top_items, 1):
            urgency_emoji = {"high": "🚨", "medium": "⚡", "low": "📝"}.get(content.urgency_level, "📝")
            relevance_bar = "█" * int(content.relevance_score * 10) + "░" * (10 - int(content.relevance_score * 10))
            
            print(f"{i:2d}. {urgency_emoji} {content.original.title[:65]}...")
            print(f"    🎯 Relevance: [{relevance_bar}] {content.relevance_score:.2f}")
            print(f"    📍 Source: {content.original.source.value} | 👤 {content.original.author}")
            print(f"    🏷️  Topics: {', '.join(content.key_topics[:4])}")
            print(f"    💡 {content.summary[:120]}...")
            print(f"    🔗 {content.original.url}")
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
