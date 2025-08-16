"""Cluster information data structures."""

from dataclasses import dataclass
from typing import List
from models import ScrapedContent


@dataclass
class ClusterInfo:
    """Information about a content cluster."""
    id: int
    items: List[ScrapedContent]
    representative_item: ScrapedContent
    theme: str
    category: str
    urgency_level: str
    size: int
