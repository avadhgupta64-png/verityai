"""VerityAI: evidence discovery and experimental article classification."""
from pathlib import Path
import json
from datetime import datetime
import requests
import streamlit as st

from src.verityai.retrieval import EvidenceRetriever, load_evidence
from src.verityai.classifier import load_classifier, classify_article

ROOT = Path(__file__).parent
SAMPLE_PATH = ROOT / "data" / "sample_evidence.jsonl"
USER_PATH = ROOT / "data" / "evidence.jsonl"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

st.set_page_config(
    page_title="VerityAI | Evidence Before Belief",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🔎 VerityAI")
st.caption("EVIDENCE BEFORE BELIEF  ·  Experimental news analysis")
st.markdown(
    "Investigate a claim by discovering relevant material and reviewing its sources. "
    "VerityAI is a research prototype: automated scores are signals for review, not verdicts."
)
with st.expander("How to interpret results", expanded=False):
    st.markdown(
        "- **Article classifier:** predicts which historical Kaggle dataset label an article resembles; it does not establish truth.\n"
        "- **Retrieval similarity:** ranks text by semantic similarity; it is not a truth probability.\n"
        "- **Evidence assessment:** remains *Unverified*. A person must inspect source quality, context, date, and conflicting evidence."
    )

classifier_tab, evidence_tab = st.tabs(["🧪 Article-pattern classifier", "🔎 Evidence discovery"])

with classifier_tab:
    st.subheader("Experimental Fake / Real article classifier")
    st.write(
        "This baseline uses TF–IDF features and logistic regression trained on the historical "
        "Kaggle ISOT Fake and Real News dataset. Publisher, topic, and time-period biases may affect results."
    )
    title = st.text_input("Article headline (optional)", key="classifier_title")
    article_text = st.text_area("Article text", height=220, key="classifier_article")
    model = load_classifier()
    if model is None:
        st.error("The trained model file was not found. Run the Kaggle training workflow and ensure models/news_classifier.joblib is available.")
    if st.button("Classify article", type="primary", disabled=(model is None or not (title.strip() or article_text.strip()))):
        try:
            result = classify_article(model, title, article_text)
            st.subheader(f"Dataset-pattern prediction: {result['label']}")
            st.metric("Model score for predicted class", f"{result['score']:.1%}")
            st.warning(
                "This score is not a calibrated probability that the article is true or false. "
                "Treat this as a dataset-pattern prediction, not a fact-check."
            )
        except Exception as exc:
            st.error(f"Classification failed: {exc}")

with evidence_tab:
    st.subheader("Find material relevant to a claim")
    st.info(
        "Live search discovers news article headlines and metadata through GDELT. "
        "It does not guarantee full-text access, source reliability, or claim verification."
    )
    claim = st.text_area(
        "Claim to investigate",
        placeholder="Write one specific, checkable claim…",
        height=100,
        key="evidence_claim",
    )
    source_mode = st.radio(
        "Evidence source",
        ["Live news discovery (GDELT)", "Local evidence corpus"],
        horizontal=True,
    )
    top_k = st.slider("Results to display", min_value=1, max_value=5, value=3)
    if source_mode == "Local evidence corpus":
        data_path = USER_PATH if USER_PATH.exists() else SAMPLE_PATH
        try:
            local_evidence = load_evidence(data_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.error(f"Could not load the evidence corpus: {exc}")
            st.stop()
        if data_path == SAMPLE_PATH:
            st.warning("Fictional demo corpus active. These passages are not real-world evidence.")
        else:
            st.caption(f"Using local corpus: {data_path.name} ({len(local_evidence)} passages)")

    @st.cache_resource(show_spinner="Loading embedding model and preparing retrieval…")
    def get_retriever(corpus_signature: str, records_json: str):
        return EvidenceRetriever(json.loads(records_json))

    def search_live_news(query: str, limit: int = 10) -> list[dict]:
        response = requests.get(
            GDELT_URL,
            params={
                "query": query,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": min(max(limit, 1), 25),
                "sort": "HybridRel",
            },
            headers={"User-Agent": "VerityAI/1.0 (research prototype)"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("articles", [])
        records = []
        for i, article in enumerate(articles):
            article_title = str(article.get("title") or "").strip()
            url = str(article.get("url") or "").strip()
            if not article_title or not url or not url.startswith(("https://", "http://")):
                continue
            seen = str(article.get("seendate") or "")
            try:
                published = datetime.strptime(seen[:8], "%Y%m%d").date().isoformat()
            except ValueError:
                published = "Date not supplied"
            domain = str(article.get("domain") or "Publisher not supplied")
            records.append({
                "id": f"gdelt-{i}",
                "title": article_title,
                "publisher": domain,
                "url": url,
                "date": published,
                "text": article_title + ". Publisher/domain: " + domain,
            })
        return records

    if st.button("Discover evidence", type="primary", disabled=not claim.strip()):
        try:
            with st.spinner("Searching and ranking relevant material…"):
                if source_mode == "Live news discovery (GDELT)":
                    records = search_live_news(claim.strip(), limit=20)
                    if not records:
                        st.warning("No article headlines were returned. Try a shorter or more specific claim.")
                        st.stop()
                    st.caption(f"Live search returned {len(records)} candidate articles. Headlines and metadata only; open sources to review the full context.")
                else:
                    records = local_evidence
                records_json = json.dumps(records, sort_keys=True, ensure_ascii=False)
                signature = str(hash(records_json))
                retriever = get_retriever(signature, records_json)
                matches = retriever.search(claim.strip(), top_k=top_k)
            st.subheader("Assessment: Unverified")
            st.write(
                "These are candidate materials ranked for relevance. Similarity does not show support or contradiction. "
                "Review the original source, publication date, evidence, and independent corroboration."
            )
            for rank, item in enumerate(matches, start=1):
                with st.container(border=True):
                    st.markdown(f"**{rank}. {item['title']}**")
                    st.caption(
                        f"{item.get('publisher', 'Unknown publisher')} · "
                        f"{item.get('date', 'Date not supplied')} · "
                        f"retrieval similarity: {item['score']:.3f}"
                    )
                    if source_mode == "Local evidence corpus":
                        st.write(item["text"])
                    else:
                        st.write("Search result headline and metadata only; the source page has not been independently assessed by VerityAI.")
                    if item.get("url"):
                        st.markdown(f"[Open original source]({item['url']})")
        except requests.RequestException as exc:
            st.error(f"Live search is temporarily unavailable. Try again later or switch to the local corpus. Details: {exc}")
        except (ValueError, KeyError, RuntimeError) as exc:
            st.error(f"Evidence discovery could not run. Details: {exc}")

st.divider()
st.caption("VerityAI · Research prototype · Verify independently before relying on a claim.")
