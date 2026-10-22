# VerityAI — Evidence Before Belief

VerityAI combines an experimental news-article classifier with a separate evidence-retrieval prototype. Classifying writing patterns is not the same as verifying a claim against independent evidence.

## Kaggle dataset

Primary dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

It contains Fake.csv (about 23.5k articles) and True.csv (about 21.4k articles), with title, text, subject, and date fields. Kaggle lists the license as CC BY-NC-SA 4.0. Review the dataset card and license before reuse; this project does not grant permission for commercial use.

### Download the dataset

1. Sign in to Kaggle and create your Kaggle API token in account settings.
2. Configure Kaggle CLI credentials using the official documentation: https://github.com/Kaggle/kaggle-api. Keep the token private; never commit it.
3. From the repository root, install dependencies and run:

    pip install -r requirements.txt
    kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw --unzip

4. Train and evaluate the baseline:

    python scripts/train_classifier.py

The script creates models/news_classifier.joblib and reports/baseline_metrics.json. The downloaded CSVs and generated model are intentionally not committed because they are large or locally generated. The app's classifier tab becomes available once the model file exists.

## Run the app

Python 3.10–3.12 is recommended.

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    streamlit run app.py

The first evidence-retrieval query downloads the configured Hugging Face embedding model and may take time.

## Two different capabilities

- Kaggle article classifier: TF-IDF + logistic regression trained on historical Fake/True labels. It estimates which dataset label an article resembles. It may learn publisher, topic, political-era, or writing-style artifacts and can be confidently wrong.
- Evidence retrieval: Hugging Face Transformers embeddings + FAISS searches a local corpus. Similarity scores rank relevance; they are not truth probabilities. The current verdict remains Unverified pending human review. It does not browse the live web.

The Kaggle dataset is a binary article-classification dataset, not a source-linked fact-checking corpus. Do not treat its labels as proof that an individual statement is objectively true or false.

## Evidence data format

Add one JSON object per line to data/evidence.jsonl. Each record should include id, title, publisher, url, date, and text. Use sources you are entitled to use and can trace. The included sample file contains fictional demonstration passages only.

## Evaluation notes

The training script reports a random stratified 80/20 split. This is a baseline, not a final benchmark: near-duplicate stories or source-specific language may inflate random-split performance. Before presenting results, add duplicate/source-aware evaluation and a held-out dataset from a different time period or source. Report macro-F1, per-class precision/recall, confusion matrix, dataset provenance, and failure examples.
