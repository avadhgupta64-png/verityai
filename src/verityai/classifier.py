"""Experimental article classifier trained on the Kaggle ISOT dataset."""
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "news_classifier.joblib"

def load_classifier():
    """Return trained local model, or None if training has not run."""
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

def classify_article(model, title: str, text: str) -> dict:
    """Predict dataset-like label; this is not fact verification."""
    combined = f"{title.strip()} {text.strip()}".strip()
    if not combined:
        raise ValueError("Enter a title or article text.")
    prediction = str(model.predict([combined])[0])
    probabilities = model.predict_proba([combined])[0]
    index = list(model.classes_).index(prediction)
    return {"label": prediction, "score": float(probabilities[index])}
