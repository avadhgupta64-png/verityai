# VerityAI

### Evidence Before Belief

VerityAI is an experimental research prototype for exploring article-writing patterns and finding material that may be relevant to a claim. It keeps two jobs separate: **article-pattern classification** and **evidence discovery**. Neither a model score nor a similarity score proves that a claim is true or false.

- **Live demo:** https://verityai19.streamlit.app/
- **Source code:** https://github.com/avadhgupta64-png/verityai

## What the app does

### 1. Article-pattern classifier

- Uses TF–IDF text features and logistic regression.
- Trained on the Kaggle Fake and Real News dataset, a historical English-language article dataset.
- Returns a dataset-style label and the model score for that predicted class.
- The score is **not** a calibrated probability of truth. Publisher, topic, time-period, and dataset-label patterns may influence predictions.

### 2. Claim review and evidence discovery

The user can choose one of two evidence sources:

- **Live news discovery:** searches candidate headlines and metadata through Google News RSS, with GDELT DOC API fallback. Results are leads for human review, not verified evidence.
- **Local evidence corpus:** retrieves passages from `data/evidence.jsonl` when supplied. If that file is absent, the app uses a clearly labelled fictional sample corpus for interface testing.

The app splits pasted text into sentence-level statements, retrieves related passages, and displays shared terms and a lexical coverage score. This is a practical first pass, not a complete linguistic claim extractor. The current implementation does not establish whether a passage supports or contradicts a claim, and it does not automatically issue a **Verified** verdict. The overall assessment remains **Unverified**.

Different percentage figures with overlapping topic terms may be flagged for review. Such a flag is only a prompt: the figures may refer to different periods, locations, definitions, or methods.

## Technology and model provenance

- **Python** — application and analysis code.
- **Streamlit** — interactive web interface.
- **scikit-learn** — TF–IDF vectorization and logistic-regression article classifier.
- **Kaggle Fake and Real News dataset** — historical labelled articles used to train the classifier.
- **Hugging Face Transformers** — loads the pretrained `sentence-transformers/all-MiniLM-L6-v2` backbone for passage embeddings.
- **FAISS** — similarity-based ranking of local passage embeddings.
- **Google News RSS and GDELT DOC API** — discovery of candidate news headlines and metadata.
- **GitHub Actions** — automated unit tests and an optional dataset-download/training workflow.

**Architecture clarification:** the article classifier is TF–IDF plus logistic regression, not a Transformer model. Transformers and FAISS are used for evidence retrieval. Retrieval similarity and lexical overlap are ranking signals, not truth probabilities.

## Run locally

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

The first evidence-retrieval request may download the pretrained embedding model and take longer than later requests. Live news discovery requires internet access. Some providers may rate-limit requests or return incomplete results; a provider failure is not evidence for or against a claim.

## Train the article classifier

1. Configure the Kaggle CLI with your own Kaggle API credentials. **Never commit credentials or tokens.**
2. Download the dataset into the expected directory:

   ```bash
   mkdir -p data/raw
   kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw --unzip
   ```

3. Train and evaluate the baseline:

   ```bash
   python scripts/train_classifier.py
   ```

The training script writes:

- `models/news_classifier.joblib` — trained pipeline.
- `reports/baseline_metrics.json` — metrics and split details.

The GitHub Actions workflow can also perform the training process. Configure the repository secrets `KAGGLE_USERNAME` and `KAGGLE_KEY`, then open **Actions → Train Kaggle baseline → Run workflow**. The workflow is designed to publish the generated model/metrics and upload an artifact. Check the latest workflow run before relying on its result.

The dataset is listed by Kaggle under CC BY-NC-SA 4.0. Review the dataset card and licence terms before reuse. This repository does not grant additional rights or permission for commercial use.

## Add a local evidence corpus

Create `data/evidence.jsonl` with one JSON object per line. Each record should include:

- `id` — stable identifier.
- `title` — passage or article title.
- `publisher` — publisher or institution.
- `url` — direct, traceable source URL.
- `date` — publication or update date when known.
- `text` — passage text.

Use material you are entitled to store and process. Keep source metadata accurate. The included `data/sample_evidence.jsonl` is fictional and must not be represented as real-world evidence.

## Tests and evaluation

Run the unit tests with:

```bash
python -m pytest -q
```

GitHub Actions is configured to run the tests on pushes to `main`, pull requests targeting `main`, and manual workflow dispatches.

`data/benchmark_15.jsonl` contains synthetic fixtures for exercising review flows. It is **not** a public-news dataset, verified ground truth, or a scored accuracy benchmark.

The current classifier evaluation uses a random stratified 80/20 split. That is only an initial baseline: duplicate stories and publisher, topic, or time-period patterns can inflate apparent performance. Stronger evaluation should use held-out sources, events, or time periods and report accuracy, macro-F1, per-class precision/recall, a confusion matrix, out-of-distribution errors, and data provenance. Evidence retrieval also needs human-labelled relevance judgements and Recall@1, Recall@3, and Recall@5. See [docs/evaluation.md](docs/evaluation.md).

## Project structure

```text
app.py                              Streamlit interface and user workflow
src/verityai/__init__.py            Package description
src/verityai/assessment.py          Statement splitting and conservative assessment
src/verityai/classifier.py          Loading and inference for the trained classifier
src/verityai/live_search.py         RSS/GDELT candidate-news discovery and parsing
src/verityai/retrieval.py           Evidence loading, embeddings, and FAISS retrieval
scripts/train_classifier.py         Training and baseline evaluation
tests/test_assessment.py            Unit tests for assessment, search, retrieval, and classifier
data/sample_evidence.jsonl          Fictional demonstration corpus
data/benchmark_15.jsonl             Synthetic review-flow fixtures
models/news_classifier.joblib       Trained model artifact
reports/baseline_metrics.json       Saved baseline evaluation metrics
requirements.txt                    Python dependencies
docs/evaluation.md                  Evaluation plan and quality gates
docs/competition-readiness.md       Competition readiness checklist
.github/workflows/tests.yml         Automated test workflow
.github/workflows/train-kaggle.yml  Optional dataset-download/training workflow
```

## Authorship, AI assistance, and attribution

The original VerityAI implementation was written by the project author without using AI tools to generate or modify the original code.

ChatGPT was used later, during a separate review revision, to suggest improvements to code readability, consolidation of duplicated logic, and project documentation. Those later review changes should be checked by the author and kept distinct from the original implementation in the Git history. No claim is made here that an external plagiarism checker has certified the project as plagiarism-free.

Third-party libraries, the pretrained embedding model, and the training dataset are external dependencies and are identified in the relevant sections above. Review their own licences and attribution requirements before distribution or competition submission.

## Limitations and responsible use

- A headline or source link alone does not establish that a claim is supported or contradicted.
- GDELT's indexed/seen date is not necessarily the publisher's original publication date.
- The included sample evidence is fictional.
- The current assessment is not a validated natural-language inference system.
- Do not use this prototype as the sole basis for consequential decisions.

Review original pages, dates, context, source quality, independent corroboration, and conflicting reporting before drawing a conclusion. Describe VerityAI as a research prototype, not an automated truth detector.
