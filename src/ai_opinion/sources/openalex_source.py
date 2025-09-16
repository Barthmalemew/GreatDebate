import requests
from datetime import datetime
from ..types import Article

BASE_URL = "https://api.openalex.org/works"

class OpenAlexSource:
    def __init__(self, query_terms, start_year: int, max_records: int = 100):
        self.query_terms = query_terms
        self.start_year = start_year
        self.max_records = max_records

    def fetch(self):
        query = " OR ".join([f'"{t}"' for t in self.query_terms])
        params = {
            "search": query,
            "filter": f"from_publication_date:{self.start_year}-01-01",
            "per-page": 200,
        }

        collected = 0
        cursor = "*"
        while collected < self.max_records:
            resp = requests.get(BASE_URL, params={**params, "cursor": cursor})
            resp.raise_for_status()
            data = resp.json()

            for rec in data.get("results", []):
                pub_date = rec.get("publication_date")
                year = int(pub_date[:4]) if pub_date else None
                if year and year < self.start_year:
                    continue

                yield Article(
                    source="OpenAlex",
                    external_id=rec.get("id"),
                    title=(rec.get("title") or "").strip(),
                    authors=[auth["author"]["display_name"] for auth in rec.get("authorships", [])],
                    abstract=(rec.get("abstract") or "").strip(),
                    url=rec.get("id"),
                    published=datetime.strptime(pub_date, "%Y-%m-%d") if pub_date else None,
                    venue=(rec.get("host_venue", {}) or {}).get("display_name"),
                )
                collected += 1
                if collected >= self.max_records:
                    break

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

