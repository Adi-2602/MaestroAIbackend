import logging
from typing import Dict, Any

logger = logging.getLogger("music-composer")

# Global variable to cache the sentiment pipeline
_sentiment_pipeline = None

def get_sentiment_analyzer():
    """
    Load the Hugging Face BERT sentiment pipeline.
    Uses distilbert-base-uncased-finetuned-sst-2-english (fast & ~268MB download).
    Falls back gracefully if import fails or offline.
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline

    try:
        from transformers import pipeline
        logger.info("Initializing Hugging Face BERT sentiment analysis pipeline...")
        # Load a standard sentiment analysis pipeline using a pre-trained DistilBERT model
        _sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        logger.info("Hugging Face BERT sentiment pipeline loaded successfully.")
    except Exception as exc:
        logger.warning(
            "Could not load Hugging Face BERT model (falling back to lexical sentiment analysis): %s",
            exc
        )
        _sentiment_pipeline = None
    return _sentiment_pipeline

def analyze_lyrics_sentiment(lyrics: list[str]) -> Dict[str, Any]:
    """
    Predict sentiment of the lyrics.
    Returns:
      dict with:
        - label: "positive" | "negative" | "neutral"
        - score: float (confidence score)
        - method: "bert" | "heuristic" | "fallback"
    """
    if not lyrics:
        return {"label": "neutral", "score": 1.0, "method": "fallback"}

    full_text = " ".join(lyrics).strip()
    if not full_text:
        return {"label": "neutral", "score": 1.0, "method": "fallback"}

    # Limit size to prevent issues with BERT token limits
    text_to_analyze = full_text[:1000]

    analyzer = get_sentiment_analyzer()
    if analyzer is not None:
        try:
            result = analyzer(text_to_analyze)[0]
            # Map standard pipeline labels ("POSITIVE"/"NEGATIVE") to lowercase
            label = result["label"].lower()
            score = float(result["score"])
            
            # Map low-confidence predictions to neutral
            if score < 0.6:
                label = "neutral"
                
            return {"label": label, "score": score, "method": "bert"}
        except Exception as exc:
            logger.error("Error running BERT sentiment analysis: %s", exc)

    # Heuristic fallback dictionary matching if transformers is not loaded
    positive_words = {"love", "happy", "joy", "smile", "dance", "bright", "dream", "sun", "light", "hope", "shine", "together", "heart", "sweet", "good", "beautiful", "glad", "celebrate"}
    negative_words = {"sad", "cry", "pain", "dark", "shadow", "rain", "hurt", "gloom", "lonely", "lost", "broken", "tears", "cold", "fear", "hate", "scared", "alone", "sorrow", "empty", "grief"}

    words = text_to_analyze.lower().split()
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)

    if pos_count > neg_count:
        label = "positive"
        score = 0.5 + 0.5 * (pos_count / (pos_count + neg_count + 1))
    elif neg_count > pos_count:
        label = "negative"
        score = 0.5 + 0.5 * (neg_count / (pos_count + neg_count + 1))
    else:
        label = "neutral"
        score = 1.0

    return {"label": label, "score": score, "method": "heuristic"}
