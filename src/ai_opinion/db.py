import sqlite3
from pathlib import Path
from typing import Iterable
from .types import Article
import json
from datetime import datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    title TEXT NOT NULL,
    authors TEXT NOT NULL, -- json list
    abstract TEXT,
    url TEXT,
    published TEXT,
    venue TEXT,
    topics TEXT, -- json list
    sentiment_compound REAL,
    relevance_score REAL, -- NEW: how relevant this is to AI sentience
    added_at TEXT NOT NULL,
    UNIQUE(source, external_id)
);
"""

class DB:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def upsert_articles(self, articles: Iterable[Article]):
        cur = self.conn.cursor()
        for a in articles:
            cur.execute(
                """
                INSERT INTO articles
                (source, external_id, title, authors, abstract, url, published, venue, topics, sentiment_compound, relevance_score, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, external_id) DO UPDATE SET
                    title=excluded.title,
                    authors=excluded.authors,
                    abstract=excluded.abstract,
                    url=excluded.url,
                    published=excluded.published,
                    venue=excluded.venue,
                    topics=excluded.topics,
                    sentiment_compound=excluded.sentiment_compound,
                    relevance_score=excluded.relevance_score,
                    added_at=excluded.added_at
                """,
                (
                    a.source,
                    a.external_id,
                    a.title,
                    json.dumps(a.authors or [], ensure_ascii=False),
                    a.abstract,
                    a.url,
                    a.published.isoformat() if a.published else None,
                    a.venue,
                    json.dumps(a.topics or [], ensure_ascii=False),
                    a.sentiment_compound,
                    a.relevance_score,
                    datetime.utcnow().isoformat(),
                ),
            )
        self.conn.commit()

    def fetch_df(self):
        import pandas as pd
        return pd.read_sql_query(
            "SELECT * FROM articles ORDER BY published DESC NULLS LAST, id DESC",
            self.conn,
        )
