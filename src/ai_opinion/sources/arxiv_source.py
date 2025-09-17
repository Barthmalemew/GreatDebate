# src/ai_opinion/sources/arxiv_source.py
import requests
import feedparser
from typing import Iterable, List
from datetime import datetime
from ..types import Article
from ..processing.nlp import clean_text

class ArxivSource:
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, query_terms: List[str], start_year: int, max_records: int = 200):
        self.query_terms = query_terms
        self.start_year = start_year
        self.max_records = max_records

    def fetch(self) -> Iterable[Article]:
        total = 0
        current_year = datetime.now().year
        terms = " OR ".join(f"\"{t}\"" for t in self.query_terms)

        for year in range(self.start_year, current_year + 1):
            if total >= self.max_records:
                break

            query = f"({terms}) AND submittedDate:[{year}01010000 TO {year}12312359]"
            url = f"{self.BASE_URL}?search_query={query}&sortBy=submittedDate&sortOrder=descending&max_results=200"

            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                parsed = feedparser.parse(r.text)
            except Exception as e:
                print(f"[WARN] arXiv fetch failed for {year}: {e}")
                continue

            if not parsed.entries:
                continue

            for entry in parsed.entries:
                if total >= self.max_records:
                    break

                published = None
                try:
                    published = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ")
                    if published.year < self.start_year:
                        continue
                except Exception:
                    pass

                yield Article(
                    source="arxiv",
                    external_id=entry.id,
                    title=entry.title.strip(),
                    authors=[a.name for a in entry.authors] if hasattr(entry, "authors") else [],
                    abstract=clean_text(entry.summary),
                    url=entry.link,
                    published=published,
                    venue="arXiv",
                )
                total += 1
