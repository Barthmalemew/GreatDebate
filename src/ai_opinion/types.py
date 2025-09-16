from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class Article:
    source: str
    external_id: str # e.g., arXiv ID or EuropePMC ID
    title: str
    authors: List[str]
    abstract: str
    url: str
    published: Optional[datetime]
    venue: Optional[str] = None
    topics: Optional[List[str]] = None
    sentiment_compound: Optional[float] = None