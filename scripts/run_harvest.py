#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
from datetime import datetime
from ai_opinion.config import QUERY_TERMS, MAX_RECORDS, DB_PATH, RELEVANCE_THRESHOLD
from ai_opinion.sources.arxiv_source import ArxivSource
from ai_opinion.sources.openalex_source import OpenAlexSource
from ai_opinion.sources.crossref_source import CrossRefSource
from ai_opinion.pipeline import collect, analyze
from ai_opinion.db import DB

def main():
    parser = argparse.ArgumentParser(
        description="Harvest and analyze AI-related scholarly items."
    )
    parser.add_argument("--query", nargs="*", help="Override query terms")
    parser.add_argument("--start-year", type=int, default=None,
                        help="Earliest publication year (default: last 5 years)")
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS)
    parser.add_argument("--db", default=DB_PATH, help="Path to SQLite DB")
    parser.add_argument("--relevance-threshold", type=float, default=RELEVANCE_THRESHOLD,
                        help="Minimum relevance score to keep an article (default from config)")
    parser.add_argument("--report", action="store_true",
                        help="Print a summary report at the end")
    args = parser.parse_args()

    # -------------------------------
    # Choose query terms
    # -------------------------------
    terms = args.query if args.query else QUERY_TERMS

    # -------------------------------
    # Start year
    # -------------------------------
    start_year = args.start_year or (datetime.now().year - 5)

    # -------------------------------
    # Sources
    # -------------------------------
    sources = [
        ArxivSource(terms, start_year=start_year, max_records=args.max_records),
        OpenAlexSource(terms, start_year=start_year, max_records=args.max_records),
        CrossRefSource(terms, start_year=start_year, max_records=args.max_records),
    ]

    # -------------------------------
    # Harvest
    # -------------------------------
    print(f"🔎 Fetching with terms: {terms} (from {start_year} onwards)")
    arts = collect(*sources)
    print(f"📥 Total collected before filtering: {len(arts)}")

    # -------------------------------
    # Analyze (includes relevance filtering)
    # -------------------------------
    arts = analyze(arts)  # analyze() internally applies relevance filtering
    # You could pass args.relevance_threshold into analyze if we modify it
    print(f"✅ Relevant articles after filtering: {len(arts)}")

    # -------------------------------
    # Store
    # -------------------------------
    db = DB(args.db)
    db.upsert_articles(arts)
    print(f"💾 Stored {len(arts)} articles into {args.db}")

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
