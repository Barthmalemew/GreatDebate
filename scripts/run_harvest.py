#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from ai_opinion.config import (
    QUERY_TERMS,
    MAX_RECORDS,
    DB_PATH,
    START_YEAR,
    ENABLE_ARXIV,
    ENABLE_OPENALEX,
    ENABLE_CROSSREF,
    ENABLE_PHILPAPERS,
    ENABLE_PSYARXIV,
)
from ai_opinion.sources.arxiv_source import ArxivSource
from ai_opinion.sources.openalex_source import OpenAlexSource
from ai_opinion.sources.crossref_source import CrossRefSource
from ai_opinion.sources.philpapers_source import PhilPapersSource
from ai_opinion.sources.psyarxiv_source import PsyArxivSource
from ai_opinion.db import DB


def fetch_source(src):
    try:
        return list(src.fetch())
    except Exception as e:
        print(f"⚠️ {src.__class__.__name__} failed: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Harvest AI-related scholarly items (no classification)."
    )
    parser.add_argument("--query", nargs="*", help="Override query terms")
    parser.add_argument("--start-year", type=int, default=None,
                        help=f"Earliest publication year (default: {START_YEAR})")
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS)
    parser.add_argument("--db", default=DB_PATH, help="Path to SQLite DB")
    parser.add_argument("--report", action="store_true",
                        help="Print a summary report at the end")
    args = parser.parse_args()

    # -------------------------------
    # Choose query terms
    # -------------------------------
    terms = args.query if args.query else QUERY_TERMS

    # -------------------------------
    # Start year (default from config)
    # -------------------------------
    start_year = args.start_year or START_YEAR

    # -------------------------------
    # Sources (respect ENABLE_* flags)
    # -------------------------------
    sources = []
    if ENABLE_ARXIV:
        sources.append(ArxivSource(terms, start_year=start_year, max_records=args.max_records))
    if ENABLE_OPENALEX:
        sources.append(OpenAlexSource(terms, start_year=start_year, max_records=args.max_records))
    if ENABLE_CROSSREF:
        sources.append(CrossRefSource(terms, start_year=start_year, max_records=args.max_records))
    if ENABLE_PHILPAPERS:
        sources.append(PhilPapersSource(terms, start_year=start_year, max_records=args.max_records))
    if ENABLE_PSYARXIV:
        sources.append(PsyArxivSource(terms, start_year=start_year, max_records=args.max_records))

    # -------------------------------
    # Parallel Harvest
    # -------------------------------
    print(f"🔎 Fetching with terms: {terms} (from {start_year} onwards)")
    articles = []
    with ThreadPoolExecutor(max_workers=len(sources)) as executor:
        futures = {executor.submit(fetch_source, s): s for s in sources}
        for future in as_completed(futures):
            src = futures[future]
            results = future.result()
            print(f"📥 {src.__class__.__name__}: {len(results)} articles")
            articles.extend(results)

    print(f"📥 Total collected: {len(articles)}")

    # -------------------------------
    # Store
    # -------------------------------
    db = DB(args.db)
    db.upsert_articles(articles)
    print(f"💾 Stored {len(articles)} articles into {args.db}")

    # -------------------------------
    # Report
    # -------------------------------
    if args.report:
        df = db.fetch_df()
        print("\n=== Harvest Report ===")
        print(f"DB size: {len(df)} articles total")
        if "published" in df.columns:
            this_year = df[df['published'].notna() & df['published'].str.startswith(str(datetime.now().year))]
            print(f"This year: {len(this_year)} new articles")
        sources_breakdown = df.groupby("source")["id"].count().to_dict()
        print(f"By source: {sources_breakdown}")
        print("======================")

if __name__ == "__main__":
    main()
