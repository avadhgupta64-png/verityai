from pathlib import Path
import json
import streamlit as st
from src.verityai.retrieval import EvidenceRetriever, load_evidence
from src.verityai.classifier import load_classifier, classify_article

ROOT = Path(__file__).parent
SAMPLE_PATH = ROOT / "data" / "sample_evidence.jsonl"
USER_PATH = ROOT / "data" / "evidence.jsonl"

st.set_page_config(page_title="VerityAI", page_icon="🔎", layout="wide")
st.title("🔎 VerityAI")
st.caption("Evidence Before Belief · Experimental news classification + evidence retrieval")
st.warning("The Kaggle-trained classifier learns patterns from historical labelled articles. It does not verify claims against current sources. Treat its output as a model prediction, not truth.")

classifier_tab, evidence_tab = st.tabs(["🧪 Kaggle article classifier", "🔎 Evidence retrieval"])
with classifier_tab:
    st.subheader("Experimental Fake / Real article classifier")
    st.write("Train locally using the Kaggle ISOT dataset. It is historical English-language news and may contain label, publisher, topic, and time-period biases.")
    title = st.text_input("Article headline (optional)")
    article_text = st.text_area("Paste article text", height=220, key="classifier_article")
    model = load_classifier()
    if model is None:
        st.info("Model not trained yet. Download the dataset, then run: python scripts/train_classifier.py")
    if st.button("Classify article", type="primary", disabled=(model is None or not (title.strip() or article_text.strip()))):
        try:
            result = classify_article(model, title, article_text)
            st.subheader(f"Dataset-pattern prediction: {result['label']}")
            st.metric("Model score for predicted class", f"{result['score']:.1%}")
            st.caption("This score is not a calibrated probability that an article is true or false. The model can fail on new events, topics, styles, and out-of-distribution text.")
        except Exception as exc:
            st.error(f"Classification failed: {exc}")

with evidence_tab:
    st.info("This tab searches a local evidence corpus only. It does not browse the live web. Similarity is not truth probability.")
    claim = st.text_area("Enter a claim to investigate", placeholder="Write one specific, checkable claim…", height=100)
    top_k = st.slider("Evidence passages to retrieve", min_value=1, max_value=5, value=3)
    data_path = USER_PATH if USER_PATH.exists() else SAMPLE_PATH
    try:
        evidence = load_evidence(data_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not load the evidence corpus: {exc}")
        st.stop()
    if data_path == SAMPLE_PATH:
        st.warning("Demo corpus active: passages are fictional examples, not real-world evidence.")
    else:
        st.caption(f"Using local corpus: {data_path.name} ({len(evidence)} passages)")

    @st.cache_resource(show_spinner="Loading embedding model and indexing evidence…")
    def get_retriever(_corpus_path: str, _corpus_signature: str):
        return EvidenceRetriever(evidence)

    if st.button("Retrieve evidence", disabled=not claim.strip()):
        try:
            signature = str(hash(tuple((x["id"], x["text"]) for x in evidence)))
            retriever = get_retriever(str(data_path), signature)
            with st.spinner("Finding passages relevant to the claim…"):
                matches = retriever.search(claim.strip(), top_k=top_k)
            st.subheader("Assessment: Unverified")
            st.write("Review each passage and source to judge support, contradiction, or insufficient evidence.")
            if not matches:
                st.warning("No passages were available to retrieve.")
            for rank, item in enumerate(matches, start=1):
                with st.container(border=True):
                    st.markdown(f"**{rank}. {item['title']}**")
                    st.caption(f"{item.get('publisher', 'Unknown publisher')} · {item.get('date', 'Date not supplied')} · retrieval similarity: {item['score']:.3f}")
                    st.write(item["text"])
                    if item.get("url"):
                        st.markdown(f"[Open source]({item['url']})")
        except Exception as exc:
            st.error(f"Evidence retrieval could not run. Check dependencies and model download access. Details: {exc}")
    with st.expander("How to add a real evidence corpus"):
        st.write("Create data/evidence.jsonl with id, title, publisher, url, date, and text fields. Use reputable, traceable sources.")
