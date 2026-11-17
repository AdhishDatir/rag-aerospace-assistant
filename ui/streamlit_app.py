"""Simple browser interface for the Aerospace RAG Assistant."""

import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Aerospace RAG Assistant", page_icon="✈️")
st.title("✈️ Aerospace RAG Assistant")
st.caption("Answers are grounded in indexed FAA technical documents.")
question = st.text_area("Ask a question", placeholder="What is the purpose of visual inspection for aircraft?", max_chars=1_000)
top_k = st.slider("Source passages to retrieve", 1, 10, 4)

if st.button("Ask documents", type="primary"):
    if not question.strip():
        st.warning("Enter a question before searching the documents.")
    else:
        with st.spinner("Searching documents and preparing a grounded answer..."):
            try:
                response = httpx.post(f"{API_BASE_URL}/ask", json={"question": question, "top_k": top_k}, timeout=120.0)
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPError as error:
                st.error("The API is unavailable. Start FastAPI, then try again.")
                st.code(str(error))
            else:
                st.subheader("Answer")
                st.write(result["answer"])
                st.caption(f"Retrieved {result['retrieval_count']} chunks in {result['latency_ms']} ms")
                st.subheader("Sources")
                if not result["citations"]:
                    st.info("No supporting source was returned for this answer.")
                for citation in result["citations"]:
                    st.markdown(f"- `{citation['document_name']}`, page {citation['page_number']}")
