# VerityAI — Evidence Before Belief

VerityAI is an evidence-retrieval prototype for examining claims. It retrieves passages from a supplied evidence corpus and helps a reviewer inspect relevant sources. **Retrieval similarity is not proof of truth**, and the app does not claim to verify the live web.

## Current scope

- Streamlit interface for entering a claim
- Hugging Face Transformers embeddings with mean pooling
- FAISS cosine-style retrieval (normalized vectors + inner product)
- Source metadata displayed alongside retrieved passages
- Conservative assessment: results remain **Unverified** until a human reviews evidence
- Explicitly illustrative sample corpus; replace it with a vetted, traceable dataset before evaluation or public use

## Quick start

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first query downloads the configured Hugging Face model, so it needs internet access and may take time. For CPU-only machines, inference can be slower.

## Evidence data format

Add one JSON object per line to `data/evidence.jsonl`:

```json
{"id":"source-001","title":"Source title","publisher":"Publisher","url":"https://example.org/source","date":"2026-01-01","text":"The evidence passage to retrieve."}
```

Only use sources you are entitled to use and can trace. Do not present illustrative data as verified facts. The included sample file is clearly marked as fictional demonstration content.

## Important limitations

- No live web search or automatic fact-checking is implemented.
- Similarity scores indicate retrieval relevance, **not** truth probability or confidence.
- The current verdict is always Unverified; a reviewer must evaluate whether sources support, contradict, or do not address the claim.
- This prototype is not a substitute for editorial review or domain expertise.

## Project structure

```
app.py
src/verityai/retrieval.py
data/sample_evidence.jsonl
tests/test_assessment.py
docs/evaluation.md
```

## Evaluation before any public claim

Use a held-out, human-labelled dataset. Report retrieval Recall@k, citation relevance/support review, macro-F1 only if a validated classifier is added, abstention rate, and latency. Document dataset provenance, licensing, language coverage, and known failure modes.
