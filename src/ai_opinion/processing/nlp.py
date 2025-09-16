from typing import List
import re


# Lightweight default sentiment: NLTK VADER
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk


# Ensure VADER lexicon is present (no-op if already downloaded)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')


_vader = SentimentIntensityAnalyzer()




def clean_text(txt: str) -> str:
    # remove references-style brackets and excessive whitespace
    txt = re.sub(r"\[[^\]]*\]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt




def sentiment_compound(text: str) -> float:
    text = clean_text(text)
    return float(_vader.polarity_scores(text).get("compound", 0.0))




def extract_keywords_corpus(texts: List[str], top_k: int = 10) -> List[List[str]]:
    """Very simple TF‑IDF keyword extraction per doc."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np


    vectorizer = TfidfVectorizer(
    max_df=0.9,
    min_df=2,
    ngram_range=(1, 2),
    stop_words='english'
    )
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()


    keywords_by_doc: List[List[str]] = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            keywords_by_doc.append([])
            continue
        idxs = row.toarray().ravel().argsort()[::-1]
        top_terms = [terms[j] for j in idxs[:top_k]]
        keywords_by_doc.append(top_terms)
    return keywords_by_doc