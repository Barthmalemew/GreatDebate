from typing import List
from .types import Article
from .processing.nlp import extract_keywords_corpus, clean_text

def analyze(articles: List[Article]) -> List[Article]:
    """
    Lightweight analysis: keyword extraction and cleaning only.
    Leaves stance/relevance to Streamlit.
    """
    texts = [clean_text((a.title or "") + ". " + (a.abstract or "")) for a in articles]
    keywords = extract_keywords_corpus(texts, top_k=8)

    analyzed: List[Article] = []
    for art, kw, text in zip(articles, keywords, texts):
        art.topics = kw
        analyzed.append(art)

    return analyzed
