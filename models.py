"""Data models for the webscraper."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum

class SourceType(Enum):
    """Enum for different content sources."""
    REDDIT = "reddit"
    STACKOVERFLOW = "stackoverflow"

@dataclass
class ScrapedContent:
    """Model for scraped content."""
    title: str
    content: str
    url: str
    source: SourceType
    timestamp: datetime
    author: str
    score: int = 0
    comments_count: int = 0
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class ProcessedContent:
    """Model for LLM-processed content."""
    original: ScrapedContent
    summary: str
    relevance_score: float
    key_topics: List[str]
    urgency_level: str  # low, medium, high

@dataclass
class DescopeInsights:
    """Model for Descope-specific business intelligence."""
    pain_points: Dict[str, float]  # pain point -> percentage
    competitive_intel: Dict[str, int]  # competitor -> opportunity count
    migration_opportunities: Dict[str, int]  # source -> prospect count
    total_discussions_analyzed: int
    high_value_opportunities: int

@dataclass
class ScrapingResult:
    """Model for complete scraping session results."""
    session_id: str
    timestamp: datetime
    total_items: int
    sources_scraped: List[SourceType]
    processed_content: List[ProcessedContent]
    overall_summary: str
    top_trends: List[str]
    descope_insights: Optional['DescopeInsights'] = None
