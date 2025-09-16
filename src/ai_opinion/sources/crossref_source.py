import requests
from datetime import datetime
from ..types import Article

BASE_URL = "https://api.crossref.org/works"

class CrossRefSource:
    """
    Fetches philosophy/psychology papers from CrossRef.
    """
    def __init__(self, query_terms, start_year: int, max_records: int = 100):
        self.query_terms = query_terms
        self.start_year = start_year
        self.max_records = max_records

    def fetch(self):
        query = " ".join(self.query_terms)
        rows = 100
        offset = 0
        collected = 0

        while collected < self.max_records:
            params = {
                "query": query,
                "filter": f"from-pub-date:{self.start_year}-01-01",
                "rows": rows,
                "offset": offset,
            }
            resp = requests.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            items = data.get("message", {}).get("items", [])
            if not items:
                break

            for rec in items:
                pub_date = rec.get("created", {}).get("date-time")
                pub = None
                if pub_date:
                    try:
                        pub = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                    except Exception:
                        pub = None

                authors = []
                for a in rec.get("author", []):
                    parts = []
                    if "given" in a: parts.append(a["given"])
                    if "family" in a: parts.append(a["family"])
                    if parts:
                        authors.append(" ".join(parts))

                yield Article(
                    source="CrossRef",
                    external_id=rec.get("DOI"),
                    title=(rec.get("title", [""])[0] if rec.get("title") else "").strip(),
                    authors=authors,
                    abstract=(rec.get("abstract") or "").strip(),
                    url=rec.get("URL"),
                    published=pub,
                    venue=rec.get("container-title", [""])[0] if rec.get("container-title") else None,
                )

                collected += 1
                if collected >= self.max_records:
                    return

            offset += rows
