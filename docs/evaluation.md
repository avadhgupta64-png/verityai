# Evaluation plan

Do not report benchmark performance until this plan is run on a documented, held-out dataset.

## Minimum evaluation

1. Select a licensed dataset with traceable sources and labels; document language, date range, collection method, and exclusions.
2. Split by source or event where feasible to reduce leakage between train and test data.
3. Manually inspect a stratified sample of retrieved passages for relevance and whether the passage actually supports the claim.
4. Report retrieval Recall@1, Recall@3, and Recall@5 where gold evidence passages exist.
5. If a separate claim classifier is later added, report per-class precision/recall, macro-F1, confusion matrix, and calibration. Do not treat cosine similarity as classifier confidence.
6. Report abstention rate, failure examples, inference latency, hardware, and model version.
7. Have reviewers distinguish support, contradiction, context mismatch, and insufficient evidence.

## Safety and quality gates

- Every displayed source must have traceable metadata.
- A source's existence or reputation alone does not prove the claim.
- Conflicting, outdated, or context-limited sources must be surfaced rather than hidden.
- The demo corpus is fictional and must never be used to label real-world claims.
- Keep the default output Unverified until an evidence-review process has been implemented and validated.
