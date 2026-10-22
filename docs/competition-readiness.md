# Competition Readiness and Verification Checklist

This checklist records the implementation state visible in the repository. It is **not a substitute for the official competition rubric**. Update it against the organizer's published criteria before submission.

## Implemented and observed working

- [x] Public Streamlit application is deployed.
- [x] Kaggle article-pattern classifier loads the trained model and returns a prediction.
- [x] Transformer embedding model loads for evidence retrieval.
- [x] FAISS returns ranked local-corpus passages.
- [x] GitHub Actions workflow downloads the Kaggle dataset, trains the baseline, saves metrics, commits the model, and uploads an artifact.
- [x] UI explains that classifier scores and retrieval similarity are not proof of truth.
- [x] Evidence assessment defaults to Unverified.
- [x] Live news discovery added using the GDELT DOC API. It surfaces candidate headlines and metadata, not a verified verdict.

## Incomplete or requiring further validation

- [ ] Obtain and map every official competition requirement, judging criterion, required plugin, and submission deliverable.
- [ ] Test live GDELT requests on the deployed Streamlit app, including API errors, empty results, and rate limits.
- [ ] Validate each returned URL, metadata field, and date interpretation; GDELT's seen/indexed date must not be assumed to be the publisher's original publication date.
- [ ] Extract and inspect full article text where access and licensing permit; headlines alone are insufficient to establish support or contradiction.
- [ ] Implement evidence reasoning that distinguishes support, contradiction, context mismatch, and insufficient evidence, with human review and conservative abstention.
- [ ] Add source-quality, recency, independence, and conflict handling without treating publisher reputation as proof.
- [ ] Run tests in CI and add unit tests for live-search parsing, empty results, invalid URLs, timeouts, and classifier behavior.
- [ ] Run source/event/time-aware and out-of-distribution evaluation; document dataset provenance, macro-F1, per-class precision/recall, confusion matrix, and failure cases.
- [ ] Evaluate retrieval using human-labelled relevance judgments and Recall@1, Recall@3, and Recall@5.
- [ ] Verify cold-start/restart behavior and dependency reproducibility on Streamlit Community Cloud.
- [ ] Review the Kaggle dataset licence and all external-source terms before public demonstration or redistribution.

## Presentation guardrails

- Describe the classifier as TF-IDF + logistic regression trained on historical Kaggle labels.
- Describe Transformers + FAISS as the retrieval stack.
- Do not describe the current system as an automated truth detector.
- Do not present fictional sample passages as evidence.
- Do not claim competition compliance until the official rubric has been checked line by line.
