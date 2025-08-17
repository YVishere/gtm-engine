"""Data models for the webscraper with enhanced RAG capabilities."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
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

# Enhanced RAG Models

@dataclass
class GitHubDiscoveryAction:
    """Enhanced action taken by RAG for repository discovery"""
    repository_name: str
    purpose: str
    relevance_score: float
    files_analyzed: List[str]
    code_snippets_found: int
    repository_stats: Dict[str, Any] = None
    analysis_summary: str = ""

    def __post_init__(self):
        if self.repository_stats is None:
            self.repository_stats = {}

@dataclass  
class EmailSolution:
    """Enhanced email solution with comprehensive metadata"""
    original_query: str
    email_content: str
    github_actions: List[GitHubDiscoveryAction]
    confidence_score: float
    generated_timestamp: str
    solution_quality: str = "medium_quality"  # low_quality, medium_quality, high_quality
    purpose_reasoning: Optional[Any] = None  # OpportunityAnalysis from enhanced_purpose_engine
    success_metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.success_metrics is None:
            self.success_metrics = {}
