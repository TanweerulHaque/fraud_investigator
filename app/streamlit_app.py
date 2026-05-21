"""Lightweight Streamlit demo for Fraud Investigator Twin."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.graph import run_investigation, seed_memory_from_file
from tools.case_loader import load_case_file, load_cases_from_json
from utils.config import get_config, resolve_path

st.set_page_config(page_title="Fraud Investigator Twin", layout="wide")
st.title("Fraud Investigator Twin")
st.caption("Digital twin PoC — reviewer judgment, not recommendations.")

cfg = get_config()

if "seeded" not in st.session_state:
    with st.spinner("Seeding historical memory..."):
        seed_memory_from_file()
    st.session_state["seeded"] = True

tab1, tab2, tab3 = st.tabs(["Investigate case", "Sample cases", "About"])

with tab1:
    source = st.radio("Case source", ["Upload JSON", "Select sample"])
    raw_case = None
    if source == "Upload JSON":
        uploaded = st.file_uploader("Case JSON", type=["json"])
        if uploaded:
            raw_case = json.load(uploaded)
    else:
        cases = load_cases_from_json(cfg["paths"]["eval_cases"])
        ids = [c["case_id"] for c in cases]
        selected = st.selectbox("Case", ids)
        raw_case = next(c for c in cases if c["case_id"] == selected)

    if st.button("Run investigation", type="primary") and raw_case:
        with st.spinner("Running LangGraph pipeline..."):
            result = run_investigation(raw_case)
        st.subheader(f"Decision: {result.decision.value}")
        st.metric("Confidence", f"{result.confidence:.0%}")
        st.markdown("### Analyst rationale")
        st.write(result.explainability.analyst_rationale)
        st.markdown("### Top reasons")
        for r in result.explainability.top_reasons:
            st.markdown(f"- {r}")
        with st.expander("Explainability (structured)"):
            st.json(result.explainability.to_analyst_sections())
        with st.expander("Agent outputs (raw + structured)"):
            st.json(result.agent_outputs)
        with st.expander("Agent trace"):
            st.json(result.agent_trace)
        with st.expander("Full result JSON"):
            st.json(json.loads(result.model_dump_json()))

with tab2:
    st.write("Cases in `data/eval_cases.json` and `data/seed_cases.json`")
    for path_key in ("seed_cases", "eval_cases"):
        st.markdown(f"**{path_key}**")
        st.json(load_cases_from_json(cfg["paths"][path_key])[:3])

with tab3:
    st.markdown(
        """
        This PoC emulates a **fraud reviewer**: inspect evidence, judge fraud/not fraud/needs review, explain why.

        It does **not** suggest customer actions. Enable LLM in `config.json` (`use_llm: true`) for optional LangChain refinement.
        """
    )
