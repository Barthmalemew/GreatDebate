import streamlit as st
import pandas as pd
import sqlite3
import json
from pathlib import Path
from datetime import datetime
import altair as alt
import re

st.set_page_config(page_title="AI Sentience Opinion Scanner", layout="wide")
st.title("🧠 AI Sentience Opinion Scanner")

# --------------------------
# Load database
# --------------------------
DB_PATH = st.sidebar.text_input("SQLite DB path", "./ai_opinion.sqlite")
if not Path(DB_PATH).exists():
    st.warning("Database not found. Run the harvester first.")
    st.stop()

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM articles", conn)
if df.empty:
    st.info("No data yet. Run the harvester.")
    st.stop()

# --------------------------
# Preprocess
# --------------------------
df = df[df["published"].notna()].copy()
df["year"] = pd.to_datetime(df["published"], errors="coerce").dt.year
current_year = datetime.now().year
df = df[df["year"].notna() & (df["year"] >= current_year - 5)]

def doc_text(row):
    return " ".join(filter(None, [str(row.get("title", "")), str(row.get("abstract", ""))]))
df["full_text"] = df.apply(doc_text, axis=1)

# --------------------------
# Stance classifier
# --------------------------
LABELS = ["Yes", "No", "Uncertain"]

# Sidebar toggle for regex
USE_REGEX = st.sidebar.checkbox("Use regex fallback (debug mode)", value=False)

classifier = None
if not USE_REGEX:
    try:
        from transformers import pipeline
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0  # use GPU (ROCm) if available
        )
        st.sidebar.success("Using HuggingFace zero-shot classifier (GPU if available)")
    except Exception as e:
        st.sidebar.error(f"Failed to load transformers pipeline: {e}")
        USE_REGEX = True

# Regex patterns for fallback
YES_PATTERNS = r"""(
    (is|are|becomes?|has|shows|demonstrates?)\s+(sentient|conscious|self[- ]aware)|
    (artificial|machine)\s+(consciousness|awareness|sentience)|
    (possesses?|exhibits?|capable of)\s+(awareness|conscious thought|subjective experience|qualia)|
    (deserves?|should be granted)\s+(personhood|moral consideration|rights)
)"""

NO_PATTERNS = r"""(
    (not|never|cannot|can't|won't|isn't|aren't)\s+(sentient|conscious|self[- ]aware)|
    lacks?\s+(sentience|consciousness|awareness)|
    no\s+(evidence|sign|proof|basis)\s+(of|for)\s+(sentience|consciousness|awareness)|
    merely\s+(a|an)\s+(tool|program|system|simulation|statistical model)|
    (just|only)\s+(an?\s+)?(algorithm|pattern recognizer|language model)|
    incapable of\s+(feeling|experience|awareness|subjectivity)|
    unfounded\s+(claims|assumptions)\s+about\s+(sentience|consciousness)
)"""

UNCERTAIN_PATTERNS = r"""(
    (might|may|could|possibly|perhaps)\s+(be|become)\s+(sentient|conscious|aware)|
    uncertain(ty)?\s+(about|regarding)?\s+(sentience|consciousness|awareness)|
    debate(s|d)?\s+(whether|if)\s+(AI|machines?)\s+(are|can be)\s+(sentient|conscious)|
    (open|ongoing)\s+question\s+(of|about)\s+(sentience|consciousness)|
    controversial\s+(topic|issue)\s+(about|regarding)\s+(AI\s+)?(sentience|consciousness)|
    unclear\s+if\s+(AI|machines?)\s+(are|can be)\s+(sentient|conscious)
)"""

def classify_text(text: str):
    if classifier:  # transformers path
        result = classifier(
            text,
            candidate_labels=LABELS,
            hypothesis_template="This text argues that AI {} be sentient."
        )
        label = result["labels"][0]
        score = float(result["scores"][0])
        return label, score

    # Regex fallback path
    text_l = text.lower()
    if re.search(NO_PATTERNS, text_l):
        return "No", 0.9
    elif re.search(YES_PATTERNS, text_l):
        return "Yes", 0.8
    elif re.search(UNCERTAIN_PATTERNS, text_l):
        return "Uncertain", 0.7
    return "Uncertain", 0.5

@st.cache_data(show_spinner=True)
def classify_all(texts):
    labels, scores = [], []
    for t in texts:
        l, s = classify_text(t)
        labels.append(l)
        scores.append(s)
    return pd.DataFrame({"stance": labels, "confidence": scores})

with st.spinner("Classifying documents..."):
    results = classify_all(df["full_text"].tolist())

df = df.reset_index(drop=True).join(results)

# --------------------------
# Sidebar filters
# --------------------------
sources = st.sidebar.multiselect("Sources", sorted(df["source"].unique()))
if sources:
    df = df[df["source"].isin(sources)]

year_min, year_max = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))
df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

query = st.sidebar.text_input("Search in title/abstract")
if query:
    ql = query.lower()
    df = df[df["title"].str.lower().str.contains(ql, na=False) | df["abstract"].str.lower().str.contains(ql, na=False)]

# --------------------------
# Overview
# --------------------------
st.subheader("Overview")
left, right = st.columns(2)
with left:
    st.metric("Articles", len(df))
with right:
    st.write("### Stance Distribution")
    stance_counts = df["stance"].value_counts().reset_index()
    stance_counts.columns = ["stance", "count"]

    pie = (
        alt.Chart(stance_counts)
        .mark_arc()
        .encode(
            theta="count:Q",
            color="stance:N",
            tooltip=["stance", "count"]
        )
        .properties(width=300, height=300)
    )

    st.altair_chart(pie, use_container_width=True)

# --------------------------
# Trends over time
# --------------------------
st.subheader("Sentience stance over time")
agg = df.groupby(["year", "stance"]).size().reset_index(name="count")
agg_total = agg.groupby("year")["count"].sum().reset_index(name="total")
agg = agg.merge(agg_total, on="year")
agg["prop"] = agg["count"] / agg["total"]

stance_order = ["Yes", "Uncertain", "No"]
agg["stance"] = pd.Categorical(agg["stance"], categories=stance_order, ordered=True)

chart = (
    alt.Chart(agg)
    .mark_line(point=True)
    .encode(
        x="year:O",
        y=alt.Y("prop:Q", title="Proportion"),
        color="stance:N",
        tooltip=["year", "stance", "prop"]
    )
    .properties(width=800, height=400)
)
st.altair_chart(chart, use_container_width=True)

# --------------------------
# Representative examples
# --------------------------
st.subheader("Representative examples")
for stance in stance_order:
    st.markdown(f"### {stance}")
    subset = df[df["stance"] == stance].sort_values("confidence", ascending=False).head(5)
    if subset.empty:
        st.caption(f"No {stance} articles found.")
        continue
    for _, row in subset.iterrows():
        st.markdown(f"- **[{row['title']}]({row['url']})** ({row['published'][:10]})")
        st.caption(f"Confidence: {row['confidence']:.2f}  \n{row['abstract'][:200]}...")

# --------------------------
# Topics / keywords
# --------------------------
st.subheader("Keyword themes")
def list_topics(x):
    try:
        arr = json.loads(x) if isinstance(x, str) else x
        return ", ".join(arr or [])
    except Exception:
        return ""
df["topic_list"] = df["topics"].apply(list_topics)
st.dataframe(df[["title", "topic_list", "stance", "year"]].sort_values("year", ascending=False).head(200), use_container_width=True)

st.caption("💡 Using HuggingFace zero-shot classification (or regex fallback if selected) to detect stance on AI sentience (Yes / No / Uncertain).")
