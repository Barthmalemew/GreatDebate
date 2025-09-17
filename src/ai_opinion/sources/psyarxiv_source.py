# src/ai_opinion/sources/psyarxiv_source.py
import requests
from typing import Iterable, List
from datetime import datetime
from ..types import Article
from ..processing.nlp import clean_text

class PsyArxivSource:
    BASE_URL = "https://api.osf.io/v2/preprints/"

    def __init__(self, query_terms: List[str], start_year: int, max_records: int = 200):
        self.query_terms = query_terms
        self.start_year = start_year
        self.max_records = max_records

    def fetch(self) -> Iterable[Article]:
        q = " OR ".join(self.query_terms)
        params = {"q": q, "page[size]": self.max_records, "provider": "psyarxiv"}
        try:
            r = requests.get(self.BASE_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"⚠️ PsyArXiv fetch failed: {e}")
            return

        for rec in data.get("data", []):
            attrs = rec.get("attributes", {})
            pub_date = attrs.get("date_published") or attrs.get("date_created")
            pub = None
            if pub_date:
                try:
                    pub = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                    if pub.year < self.start_year:
                        continue
                except:
                    pub = None

            yield Article(
                source="psyarxiv",
                external_id=rec.get("id", ""),
                title=attrs.get("title", "").strip(),
                authors=[],
                abstract=clean_text(attrs.get("description", "")),
                url=attrs.get("doi") or attrs.get("links", {}).get("html"),
                published=pub,
                venue="PsyArXiv",
            )
