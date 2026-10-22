"""Load and run the experimental historical article-pattern classifier."""
from pathlib import Path
from typing import Any

import joblib

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "news_classifier.joblib"


def load_classifier() -> Any | None:
    """Load the trained pipeline, or return None when the model file is absent."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def classify_article(model: Any, title: str, text: str) -> dict[str, str | float]:
    """Predict a dataset-style label and its score, not the truth of an article.

    The model's class score is not calibrated as a probability that the article
    is true or false. This function only exposes the trained model's prediction.
    """
    combined_text = f"{title.strip()} {text.strip()}".strip()
    if not combined_text:
        raise ValueError("Enter a title or article text.")

    prediction = str(model.predict([combined_text])[0])
    class_scores = model.predict_proba([combined_text])[0]
    predicted_index = list(model.classes_).index(prediction)

    return {
        "label": prediction,
        "score": float(class_scores[predicted_index]),
    }
