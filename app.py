from pathlib import Path
import json

import streamlit as st

from src.verityai.retrieval import EvidenceRetriever, load_evidence

ROOT = Path(__file__).parent
SAMPLE_PATH = ROOT / "data" / "sample_evidence.jsonl"
USER_PATH = ROOT / "data" / "evidence.jsonl"

st.set_page_config(page_title="VerityAI", page_icon="🔎", layout="wide")
st.title("🔎 VerityAI")
st.caption("Evidence Before Belief · Evidence retrieval, not an automatic truth oracle")

st.info(
    "This prototype searches only its local evidence corpus. It does not browse the live web. "
    "Similarity scores measure retrieval relevance, not whether a claim is true."
)

claim = st.text_area(
    "Enter a claim to investigate",
    placeholder="Write one specific, checkable claim…",
    height=110,
)
top_k = st.slider("Evidence passages to retrieve", min_value=1, max_value=5, value=3)

data_path = USER_PATH if USER_PATH.exists() else SAMPLE_PATH
try:
    evidence = load_evidence(data_path)
except (OSError, ValueError, json.JSONDecodeError) as exc:
    st.error(f"Could not load the evidence corpus: {exc}")
    st.stop()

if data_path == SAMPLE_PATH:
    st.warning("Demo corpus active: its passages are fictional examples, not real-world evidence.")
else:
    st.caption(f"Using local corpus: {data_path.name} ({len(evidence)} passages)")

@st.cache_resource(show_spinner="Loading embedding model and indexing evidence…")
def get_retriever(_corpus_path: str, _corpus_signature: str):
    # Signature changes when corpus text changes, invalidating the cached index.
    return EvidenceRetriever(evidence)

if st.button("Retrieve evidence", type="primary", disabled=not claim.strip()):
    try:
        signature = str(hash(tuple((x["id"], x["text"]) for x in evidence)))
        retriever = get_retriever(str(data_path), signature)
        with st.spinner("Finding passages relevant to the claim…"):
            matches = retriever.search(claim.strip(), top_k=top_k)
        st.subheader("Assessment: Unverified")
        st.write(
            "No truth verdict is generated automatically. Review each passage and its source "
            "to decide whether it supports, contradicts, or does not address the claim."
        )
        if not matches:
            st.warning("No passages were available to retrieve.")
        for rank, item in enumerate(matches, start=1):
            with st.container(border=True):
                st.markdown(f"**{rank}. {item['title']}**")
                st.caption(
                    f"{item.get('publisher', 'Unknown publisher')} · "
                    f"{item.get('date', 'Date not supplied')} · "
                    f"retrieval similarity: {item['score']:.3f}"
                )
                st.write(item["text"])
                url = item.get("url", "")
                if url:
                    st.markdown(f"[Open source]({url})")
    except Exception as exc:
        st.error(
            "Evidence retrieval could not run. Check that dependencies are installed and "
            "the Hugging Face model can be downloaded. Details: " + str(exc)
        )

with st.expander("How to add a real evidence corpus"):
    st.write(
        "Create data/evidence.jsonl with one JSON object per line. Each record should include "
        "id, title, publisher, url, date, and text. Use reputable, traceable sources and "
        "document licensing and collection methods before evaluating the system."
    )
