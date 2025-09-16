from transformers import pipeline

# Load HuggingFace model once
_zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def sentience_relevance(text: str):
    """
    Returns (is_relevant: bool, score: float) for AI sentience relevance.
    """
    if not text or not text.strip():
        return False, 0.0

    labels = ["Relevant to AI sentience", "Not relevant"]
    result = _zero_shot(text, candidate_labels=labels, multi_label=False)

    top_label = result["labels"][0]
    top_score = result["scores"][0]

    return (top_label == "Relevant to AI sentience", float(top_score))
