# src/ai_opinion/config.py

# --- Query terms ---
# Broad search terms for harvesting AI-related papers
QUERY_TERMS = [
    "artificial intelligence",
    "AI"
]

# --- Harvest limits ---
MAX_RECORDS = 500
START_YEAR = 2020  # Default lower bound for publication year

# --- Database ---
DB_PATH = "./ai_opinion.sqlite"

# --- Sources (toggles) ---
ENABLE_ARXIV = True
ENABLE_OPENALEX = True
ENABLE_CROSSREF = True
ENABLE_PSYARXIV = True
