# src/ai_opinion/db.py
import sqlite3
import json
from datetime import datetime
from typing import Iterable
from .types import Article

class DB:
    def __init__(self, path: str = "ai_opinion.sqlite"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.create_schema()

    def create_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            source TEXT,
            external_id TEXT,
            title TEXT,
            authors TEXT,
            abstract TEXT,
            url TEXT,
            published TEXT,
            venue TEXT,
            topics TEXT,
            sentiment_compound REAL,
            added_at TEXT,
            UNIQUE(source, external_id)
        )
        """)
        self.conn.commit()

    def upsert_articles(self, articles: Iterable[Article]):
        cur = self.conn.cursor()
        sql = """
            INSERT OR IGNORE INTO articles
            (source, external_id, title, authors, abstract, url, published, venue, topics, sentiment_compound, added_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        for a in articles:
            cur.execute(
                sql,
                (
                    a.source,
                    a.external_id,
                    a.title,
                    json.dumps(a.authors, ensure_ascii=False),
                    a.abstract,
                    a.url,
                    a.published.isoformat() if getattr(a, "published", None) else None,
                    getattr(a, "venue", None),
                    json.dumps(getattr(a, "topics", []) or [], ensure_ascii=False),
                    getattr(a, "sentiment_compound", None),
                    datetime.utcnow().isoformat(),
                ),
            )
        self.conn.commit()

    def fetch_df(self):
        import pandas as pd
        return pd.read_sql_query("SELECT * FROM articles", self.conn)
