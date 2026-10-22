"""Train and evaluate the TF-IDF/logistic-regression article baseline."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def read_articles(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Download the Kaggle data using README instructions.")
    # Keep only the two fields used by the baseline and normalise missing values.
    frame = pd.read_csv(path)
    if not {"title", "text"}.issubset(frame.columns):
        raise ValueError(f"{path.name} must contain title and text columns.")
    frame = frame[["title", "text"]].fillna("")
    frame["content"] = (frame["title"].astype(str) + " " + frame["text"].astype(str)).str.strip()
    frame = frame[frame["content"].str.len() > 0]
    frame["label"] = label
    return frame[["content", "label"]]

def main() -> None:\n    """Train the baseline, print evaluation metrics, and save reproducible outputs."""
    fake = read_articles(RAW / "Fake.csv", "Fake")
    real = read_articles(RAW / "True.csv", "Real")
    data = pd.concat([fake, real], ignore_index=True).drop_duplicates("content")
    x_train, x_test, y_train, y_test = train_test_split(
        data["content"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", stop_words="english",
            max_features=250000, ngram_range=(1, 2), min_df=2, max_df=0.98,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    print(f"Training on {len(x_train):,} articles; testing on {len(x_test):,}.")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "dataset": "Kaggle ISOT Fake and Real News Dataset",
        "split": "random stratified 80/20; random_state=42",
        "train_rows": int(len(x_train)), "test_rows": int(len(x_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "confusion_matrix_labels": ["Fake", "Real"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["Fake", "Real"]).tolist(),
        "classification_report": report,
        "caveat": "In-distribution benchmark only; not proof of real-world truth verification."
    }
    model_dir, report_dir = ROOT / "models", ROOT / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "news_classifier.joblib")
    (report_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Saved model: {model_dir / 'news_classifier.joblib'}")
    print(f"Saved metrics: {report_dir / 'baseline_metrics.json'}")

if __name__ == "__main__":
    main()
