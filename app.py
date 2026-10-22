from pathlib import Path
import hashlib
import json

import requests
import streamlit as st

from src.verityai.assessment import (
    assess_facts,
    find_numeric_discrepancies,
    split_claim_into_facts,
)
from src.verityai.classifier import classify_article, load_classifier
from src.verityai.live_search import search_live_news
from src.verityai.retrieval import EvidenceRetriever, load_evidence


# Paths are resolved relative to this file so the app can run from any working directory.
ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "sample_evidence.jsonl"
USER_PATH = ROOT / "data" / "evidence.jsonl"

st.set_page_config(
    page_title="VerityAI | Evidence Before Belief",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("VerityAI")
st.caption("EVIDENCE BEFORE BELIEF  |  Misinformation and media-literacy research")
st.write(
    "Break a news excerpt into smaller factual statements, find related material, "
    "and inspect the sources yourself. VerityAI is a research prototype, not an authority on truth."
)

with st.expander("How results are assessed"):
    st.markdown(
        "- **Verified** should require strong, attributable evidence and appropriate review. This prototype deliberately does not issue that label automatically.\n"
        "- **Potentially Misleading** is a caution flag when a possible textual contradiction needs review; it is not a definitive finding.\n"
        "- **Unverified** means the available material does not establish the statement. It does not mean the statement is false.\n"
        "- The coverage score describes word overlap with the closest retrieved passage. It is not a probability of truth."
    )

# Streamlit tabs reset to the first tab after a rerun. Put Claim analysis first so
# results and diagnostics remain visible after its Analyse claim button reruns the app.
analysis_tab, classifier_tab = st.tabs(["Claim analysis", "Article-pattern classifier"])

with classifier_tab:
    st.subheader("Historical article-pattern baseline")
    st.write(
        "A TF–IDF and logistic-regression model trained on the Kaggle Fake and Real News dataset. "
        "It predicts dataset-style labels and cannot verify a current claim."
    )
    headline = st.text_input("Headline (optional)")
    article = st.text_area("Article text", height=200)
    model = load_classifier()
    if model is None:
        st.error("Trained model not found. Run the training workflow and ensure the model file is available.")
    if st.button("Classify article", type="primary", disabled=model is None or not (headline.strip() or article.strip())):
        try:
            result = classify_article(model, headline, article)
            st.subheader(f"Dataset-pattern prediction: {result['label']}")
            st.metric("Score for predicted class", f"{result['score']:.1%}")
            st.warning("This is not a calibrated probability that the article is true or false.")
        except (ValueError, AttributeError, TypeError) as exc:
            st.error(f"Could not classify this input: {exc}")

with analysis_tab:
    st.subheader("Break down and investigate a claim")
    claim = st.text_area(
        "Paste a claim or news excerpt",
        placeholder="Example: The report was published on Tuesday. It says the city opened two new clinics.",
        height=150,
    )
    source_choice = st.radio(
        "Where should VerityAI look?",
        ["Live news headlines (GDELT)", "Local evidence corpus"],
        horizontal=True,
    )
    result_count = st.slider("Maximum passages per statement", 1, 5, 3)
    if source_choice == "Local evidence corpus":
        corpus_path = USER_PATH if USER_PATH.exists() else SAMPLE_PATH
        try:
            local_records = load_evidence(corpus_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.error(f"The evidence file could not be loaded: {exc}")
            st.stop()
        if corpus_path == SAMPLE_PATH:
            st.warning("The included sample corpus is fictional. Use it only to test the interface, never as real evidence.")
        else:
            st.caption(f"Loaded {len(local_records)} passages from {corpus_path.name}.")

    @st.cache_resource(show_spinner="Loading the embedding model…")
    def build_retriever(cache_key: str, records_json: str):
        return EvidenceRetriever(json.loads(records_json))

    if st.button("Analyse claim", type="primary", disabled=not claim.strip()):
        all_facts = split_claim_into_facts(claim)
        if not all_facts:
            st.warning("Enter at least one factual statement.")
            st.stop()
        max_statements = 12
        if len(all_facts) > max_statements:
            st.warning(
                f"This input contains {len(all_facts)} sentence-level statements. "
                f"To keep analysis responsive, only the first {max_statements} will be analysed. "
                "Shorten the input or submit the remaining statements separately."
            )
        facts = all_facts[:max_statements]
        st.markdown("#### 1. Atomic statements")
        for number, fact in enumerate(facts, start=1):
            st.write(f"{number}. {fact}")

        try:
            with st.spinner("Finding relevant material for each statement…"):
                if source_choice == "Live news headlines (GDELT)":
                    # Search each statement separately: a long combined paragraph
                    # often produces poor search results or no results at all.
                    live_records = []
                    live_search_failures = []
                    for fact in facts:
                        try:
                            live_records.extend(
                                search_live_news(fact, limit=max(10, result_count * 3))
                            )
                        except (requests.RequestException, ValueError) as exc:
                            # GDELT can return HTML/plain text with HTTP 200;
                            # search_live_news reports that as ValueError. Treat it
                            # as a per-statement live-search failure, not a fatal
                            # failure of the whole claim-analysis pipeline.
                            live_search_failures.append(
                                f"{fact}: {type(exc).__name__}: {str(exc)[:240]}"
                            )
                    if live_search_failures:
                        if live_records:
                            st.warning(
                                f"Live news search failed for {len(live_search_failures)} of "
                                f"{len(facts)} statements. Results below use only successful "
                                "live searches; failed statements are not backed by live results. "
                                "No local/demo evidence was substituted."
                            )
                        else:
                            st.error(
                                "Live news search failed for every statement. VerityAI tried its primary "
                                "provider (GDELT) and the Google News RSS fallback, but neither produced "
                                "usable results. This is a provider/network failure, not a finding about "
                                "whether the claim is true. No local/demo evidence was substituted."
                            )
                            with st.expander("Live-search diagnostics (provider errors)"):
                                st.caption(
                                    "Each line should include both the GDELT error and the Google News RSS "
                                    "fallback error. If it mentions only GDELT, the deployed app may be "
                                    "running an older version of live_search.py."
                                )
                                for failure in live_search_failures:
                                    st.code(failure)
                            st.info(
                                "If this persists, open Manage app → Cloud logs and check whether the latest "
                                "GitHub commit has been deployed. Streamlit Community Cloud may need a manual reboot."
                            )
                            st.stop()
                    unique_records = {}
                    for item in live_records:
                        identity = item.get("url") or item.get("id") or item.get("title")
                        unique_records[identity] = item
                    records = list(unique_records.values())
                    if not records:
                        st.warning("GDELT returned no candidate headlines for these statements. No local/demo evidence was substituted. Try shorter wording or switch to the local corpus.")
                        st.stop()
                    st.caption(f"Found {len(records)} unique candidate headlines across {len(facts)} statement-specific searches. Headlines and metadata alone may not contain enough evidence.")
                else:
                    records = local_records

                records_json = json.dumps(records, sort_keys=True, ensure_ascii=False)
                cache_key = hashlib.sha256(records_json.encode("utf-8")).hexdigest()
                retriever = build_retriever(cache_key, records_json)
                per_fact_evidence = []
                for fact in facts:
                    per_fact_evidence.extend(retriever.search(fact, top_k=result_count))
                unique = {}
                for item in per_fact_evidence:
                    identity = item.get("url") or item.get("id") or item.get("title")
                    if identity not in unique or item["score"] > unique[identity]["score"]:
                        unique[identity] = item
                evidence = sorted(unique.values(), key=lambda item: item["score"], reverse=True)
                assessments = assess_facts(facts, evidence)
                discrepancies = find_numeric_discrepancies(facts)

            st.markdown("#### 2. Evidence assessment")
            st.info(
                "Overall status: **Unverified**. The current assessment uses retrieval and transparent word-overlap signals. "
                "It does not yet have a validated natural-language inference model or authoritative-source review."
            )
            st.caption("Evidence coverage score is based on shared terms with a sufficiently relevant passage. It is not a truth probability or calibrated confidence score.")
            if discrepancies:
                st.markdown("#### Possible numerical discrepancies to review")
                st.warning("Different figures may refer to different locations, periods, definitions, or methods. These are prompts for human review, not automatic contradiction findings.")
                for discrepancy in discrepancies:
                    st.write(f"**Figures:** {', '.join(discrepancy['figures'])}")
                    st.write(discrepancy["statement_a"])
                    st.write(discrepancy["statement_b"])
                    st.caption(discrepancy["note"])
            for number, item in enumerate(assessments, start=1):
                with st.container(border=True):
                    st.markdown(f"**Statement {number}: {item['fact']}**")
                    st.write(f"Status: **{item['status']}**")
                    st.progress(float(item["coverage_score"]), text=f"Textual evidence coverage: {item['coverage_score']:.0%}")
                    st.write(item["reason"])
                    if item["matched_terms"]:
                        st.caption("Shared terms: " + ", ".join(item["matched_terms"]))
                    source = item.get("source")
                    if source:
                        st.markdown(f"**Closest retrieved item:** {source.get('title', 'Untitled source')}")
                        st.caption(
                            f"{source.get('publisher', 'Publisher not supplied')} · "
                            f"{source.get('date_label', 'Date')} {source.get('date', 'not supplied')} · "
                            f"retrieval similarity {source.get('score', 0):.3f}"
                        )
                        if source.get("url"):
                            st.markdown(f"[Open source]({source['url']})")
                        if source_choice == "Local evidence corpus":
                            st.write(source.get("text", ""))
                    else:
                        st.write("No matching source was retrieved for this statement.")

            log = {
                "claim": claim,
                "atomic_statements": facts,
                "overall_status": "Unverified",
                "method": "Transformer embeddings + FAISS retrieval + lexical relevance heuristic",
                "limitations": [
                    "No automatic truth determination",
                    "GDELT mode searches each statement separately and returns headlines/metadata, not full article content",
                    "Weak lexical matches are omitted as source matches",
                    "Different percentages may be flagged for human review but are not automatically treated as contradictions",
                    "Lexical overlap is not entailment",
                    "The included demo corpus is fictional",
                ],
                "possible_numeric_discrepancies": discrepancies,
                "assessments": [
                    {
                        "fact": item["fact"],
                        "status": item["status"],
                        "coverage_score": item["coverage_score"],
                        "matched_terms": item["matched_terms"],
                        "reason": item["reason"],
                        "source": item["source"],
                    }
                    for item in assessments
                ],
            }
            st.download_button(
                "Download source-grounded analysis log (JSON)",
                data=json.dumps(log, ensure_ascii=False, indent=2),
                file_name="verityai-analysis-log.json",
                mime="application/json",
            )
            st.markdown("#### 3. Retrieved sources")
            if not evidence:
                st.write("No source candidates were available.")
            for rank, item in enumerate(evidence[:10], start=1):
                with st.container(border=True):
                    st.markdown(f"**{rank}. {item.get('title', 'Untitled source')}**")
                    st.caption(
                        f"{item.get('publisher', 'Publisher not supplied')} · "
                        f"{item.get('date_label', 'Date')} {item.get('date', 'not supplied')} · "
                        f"similarity {item.get('score', 0):.3f}"
                    )
                    if source_choice == "Local evidence corpus":
                        st.write(item.get("text", ""))
                    else:
                        st.write("Headline and metadata only; review the original page before using it as evidence.")
                    if item.get("url"):
                        st.markdown(f"[Open original source]({item['url']})")
        except requests.RequestException:
            st.error("The live news service could not be reached. Try again later or switch to the local corpus.")
        except (ValueError, KeyError, RuntimeError) as exc:
            st.error(f"Claim analysis could not be completed: {exc}")

st.divider()
st.caption("VerityAI · Research prototype · Check original sources and context before reaching a conclusion.")
