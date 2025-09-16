from typing import List
from .types import Article
from .processing.nlp import sentiment_compound, extract_keywords_corpus, clean_text
from .relevance import sentience_relevance

def analyze(articles: List[Article]) -> List[Article]:
    texts = [clean_text((a.title or "") + ". " + (a.abstract or "")) for a in articles]
    keywords = extract_keywords_corpus(texts, top_k=8)

    analyzed: List[Article] = []
    for art, kw, text in zip(articles, keywords, texts):
        # Sentience relevance filter
        is_rel, rel_score = sentience_relevance(text)
        art.relevance_score = rel_score
        if not is_rel:
            continue  # skip irrelevant papers

        # Sentiment + keywords
        art.sentiment_compound = sentiment_compound(text)
        art.topics = kw
        analyzed.append(art)

    return analyzed

def collect(*sources) -> List[Article]:
    out: List[Article] = []
    for src in sources:
        for art in src.fetch():
            out.append(art)
    return out
