from typing import Iterable, List
from datetime import datetime
import arxiv
from ..types import Article


import arxiv
from arxiv import UnexpectedEmptyPageError
from typing import Iterable, List
from datetime import datetime
from ..types import Article

class ArxivSource:
    def __init__(self, query_terms: List[str], start_year: int, max_records: int = 100):
        self.query_terms = query_terms
        self.start_year = start_year
        self.max_records = max_records

    def fetch(self) -> Iterable[Article]:
        query = " OR ".join([f'"{q}"' for q in self.query_terms])
        search = arxiv.Search(
            query=query,
            max_results=self.max_records,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        client = arxiv.Client()
        try:
            for r in client.results(search):
                pub = r.published.replace(tzinfo=None) if isinstance(r.published, datetime) else None
                if pub and pub.year < self.start_year:
                    continue
                yield Article(
                    source="arxiv",
                    external_id=r.get_short_id(),
                    title=r.title.strip(),
                    authors=[a.name for a in r.authors],
                    abstract=r.summary.strip(),
                    url=r.entry_id,
                    published=pub,
                    venue=", ".join(r.categories) if r.categories else None,
                )
        except UnexpectedEmptyPageError as e:
            print(f"[WARN] arXiv returned empty page for query {query}: {e}")
            return
