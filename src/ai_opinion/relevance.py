from transformers import pipeline

_zero_shot = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0,         # use GPU
    batch_size=32     # process in batches
)

def sentience_relevance_batch(texts: list[str]):
    """
    Batched relevance scoring for a list of texts.
    Returns list of (is_relevant: bool, score: float).
    """
    if not texts:
        return []

    labels = ["Relevant to AI sentience", "Not relevant"]

    results = _zero_shot(
        texts,
        candidate_labels=labels,
        multi_label=False
    )

    # HF quirk: single input returns dict instead of list
    if isinstance(results, dict):
        results = [results]

    out = []
    for res in results:
        top_label = res["labels"][0]
        top_score = res["scores"][0]
        out.append((top_label == "Relevant to AI sentience", float(top_score)))
    return out
