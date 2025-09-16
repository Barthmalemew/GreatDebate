#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import sqlite3
import json
from pathlib import Path
from datetime import datetime
import altair as alt
import re

st.set_page_config(page_title="Great Debate", layout="wide")
st.title("🧠 Great Debate")

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
# Classifier setup
# --------------------------
LABELS = ["Yes", "No", "Uncertain"]

USE_REGEX = st.sidebar.checkbox("Use regex fallback only", value=False)

classifier = None
if not USE_REGEX:
    try:
        from transformers import pipeline
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0  # use GPU if available
        )
        st.sidebar.success("Using HuggingFace zero-shot classifier (GPU if available)")
    except Exception as e:
        st.sidebar.error(f"Failed to load transformers pipeline: {e}")
        USE_REGEX = True

# --------------------------
# Regex patterns
# --------------------------
YES_PATTERNS = r"""(
    (is|are|becomes?|has|shows|demonstrates?)\s+(sentient|conscious|self[- ]aware)|
    (artificial|machine)\s+(consciousness|awareness|sentience)|
    (possesses?|exhibits?|capable of)\s+(awareness|conscious thought|subjective experience|qualia)|
    (deserves?|should be granted)\s+(personhood|moral consideration|rights)
)"""

NO_PATTERNS = r"""(
    (not|never|cannot|can't|won't|isn't|aren't)\s+(sentient|conscious|self[- ]aware)|
    (does\s+not|fails?\s+to|unlikely\s+to)\s+(show|exhibit|possess)\s+(consciousness|awareness|sentience)|
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

# --------------------------
# Unified batch classifier
# --------------------------
# --------------------------
# Unified batch classifier (improved)
# --------------------------
@st.cache_data(show_spinner=True)
def classify_all(texts, batch_size: int = 8):
    labels, scores = [], []

    if classifier:  # HuggingFace path
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]

            results = classifier(
                batch,
                candidate_labels=["Yes", "No", "Uncertain"],
                hypothesis_template="This text suggests that AI sentience is {}.",
                truncation=True
            )
            if isinstance(results, dict):
                results = [results]

            for text, res in zip(batch, results):
                text_l = text.lower()
                scores_dict = {lab: float(sc) for lab, sc in zip(res["labels"], res["scores"])}

                # regex boosters
                if re.search(NO_PATTERNS, text_l):
                    scores_dict["No"] += 0.25
                if re.search(YES_PATTERNS, text_l):
                    scores_dict["Yes"] += 0.2
                if re.search(UNCERTAIN_PATTERNS, text_l):
                    scores_dict["Uncertain"] += 0.05

                # normalize
                total_score = sum(scores_dict.values())
                for k in scores_dict:
                    scores_dict[k] /= total_score

                # sorted scores
                sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
                best, best_score = sorted_scores[0]
                second, second_score = sorted_scores[1]

                # --- Shrink "Uncertain" logic ---
                if best_score < 0.4:
                    mapped, final_score = "Uncertain", best_score
                elif best == "Uncertain":
                    if abs(scores_dict["Yes"] - scores_dict["No"]) < 0.15:
                        # close call → choose whichever is higher
                        mapped = "Yes" if scores_dict["Yes"] >= scores_dict["No"] else "No"
                        final_score = max(scores_dict["Yes"], scores_dict["No"])
                    else:
                        mapped, final_score = "Uncertain", best_score
                else:
                    mapped, final_score = best, best_score

                labels.append(mapped)
                scores.append(final_score)

    else:  # Regex-only fallback
        for text in texts:
            text_l = text.lower()
            if re.search(NO_PATTERNS, text_l):
                labels.append("No"); scores.append(0.85)
            elif re.search(YES_PATTERNS, text_l):
                labels.append("Yes"); scores.append(0.75)
            elif re.search(UNCERTAIN_PATTERNS, text_l):
                labels.append("Uncertain"); scores.append(0.6)
            else:
                labels.append("No"); scores.append(0.55)  # default lean No

    return pd.DataFrame({"stance": labels, "confidence": scores})


# --------------------------
# Run classification
# --------------------------
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
st.dataframe(
    df[["title", "topic_list", "stance", "year"]]
    .sort_values("year", ascending=False)
    .head(200),
    use_container_width=True
)

st.caption("💡 Using HuggingFace zero-shot classification with regex safeguards (or regex fallback) to detect stance on AI sentience (Yes / No / Uncertain).")
