# VerityAI
### Evidence Before Belief

VerityAI is an experimental research prototype for exploring news-article patterns and discovering material relevant to a claim. It deliberately separates **article-pattern classification** from **evidence discovery**: neither a model score nor a similarity score proves that a claim is true or false.

**Live demo:** https://verityai19.streamlit.app/  
**Repository:** https://github.com/avadhgupta64-png/verityai

## Capabilities

### 1. Article-pattern classifier
- TF–IDF text features with logistic regression.
- Trained on the Kaggle Fake and Real News dataset (historical English-language articles).
- Reports the predicted dataset-style label and the model's score for that class.
- The score is not a calibrated probability of truth. Dataset labels, publishers, topics, and historical language patterns can introduce bias.

### 2. Evidence discovery
Choose one of two sources:
- **Live news discovery (GDELT):** queries the GDELT DOC API for candidate article headlines and metadata, then ranks candidates with embeddings. It does not guarantee full-text access or assess whether a source actually supports or contradicts the claim.
- **Local evidence corpus:** retrieves passages from `data/evidence.jsonl`, if supplied. Otherwise, the app clearly identifies the included fictional demonstration corpus.

The assessment intentionally remains **Unverified**. Review original pages, dates, context, source quality, independent corroboration, and conflicting reporting before drawing a conclusion.

## Technology

- Python
- Streamlit
- scikit-learn: TF–IDF + logistic regression classifier
- Hugging Face Transformers: `sentence-transformers/all-MiniLM-L6-v2` backbone for passage embeddings
- FAISS: vector-similarity retrieval
- Kaggle: historical article dataset
- GDELT DOC API: live news-article discovery
- GitHub Actions: repeatable dataset download, training, evaluation, model artifact publication, and unit-test CI

**Architecture note:** the article classifier is not a Transformer model. Transformers and FAISS are used in the evidence-retrieval path.

## Run locally

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The first evidence-discovery request may download the Hugging Face embedding model and can take longer than later requests. Live news discovery requires outbound access to the GDELT API. Pushes and pull requests to `main` run the automated test suite through GitHub Actions.

## Train the article classifier

1. Create Kaggle API credentials and configure the Kaggle CLI. Never commit credentials.
2. Download the dataset:
   ```bash
   mkdir -p data/raw
   kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw --unzip
   ```
3. Train and evaluate:
   ```bash
   python scripts/train_classifier.py
   ```

The training script creates:
- `models/news_classifier.joblib`
- `reports/baseline_metrics.json`

The GitHub Actions workflow can perform this process without a local Python environment. Configure repository Actions secrets `KAGGLE_USERNAME` and `KAGGLE_KEY`, then open **Actions → Train Kaggle baseline → Run workflow**. On success, the workflow commits the trained model and metrics and uploads a backup artifact.

The dataset is listed by Kaggle under CC BY-NC-SA 4.0. Review the dataset card and licence terms before reuse; this repository does not grant additional rights or permission for commercial use.

## Add a local evidence corpus

Create `data/evidence.jsonl` with one JSON object per line. Each record should include:
- `id`: stable identifier
- `title`: passage or article title
- `publisher`: publisher or institution
- `url`: direct, traceable source URL
- `date`: publication or update date when known
- `text`: passage text

Only add material you are entitled to use. Keep source metadata accurate. The included `data/sample_evidence.jsonl` is fictional and must not be presented as real-world evidence.

## Evaluation and limitations

The current classifier uses a random stratified 80/20 split. That is a baseline, not proof of generalization; duplicate stories, publisher language, and topic/time-period artifacts can inflate performance. Before making benchmark claims:
- evaluate on a held-out source, event, or time period;
- report accuracy, macro-F1, per-class precision/recall, and the confusion matrix;
- inspect errors and test out-of-distribution examples;
- evaluate evidence retrieval with human-labelled relevance judgments and Recall@1, Recall@3, and Recall@5;
- document latency, model version, data provenance, and failure cases.

Live search results are candidate sources, not verified evidence. GDELT metadata and headlines can be incomplete or noisy, and retrieval similarity is not a truth score. VerityAI does not currently automatically classify evidence as supporting or contradicting a claim. The final assessment remains **Unverified** pending human review.

## Project structure

```text
app.py                         Streamlit interface
src/verityai/classifier.py     Classifier loading and inference
src/verityai/retrieval.py      Transformer embeddings and FAISS retrieval
scripts/train_classifier.py    Training and baseline evaluation
data/sample_evidence.jsonl     Fictional demo corpus
docs/evaluation.md             Evaluation protocol
.github/workflows/tests.yml    Continuous integration tests\ntests/test_assessment.py       Evidence-loader, live-search parser, and classifier tests
.github/workflows/             Training workflow
```

## Responsible use

Do not use this prototype as the sole basis for consequential decisions. Verify claims using original, authoritative, current, and independent sources. Do not describe a prediction as proof or a similarity score as confidence in truth.
