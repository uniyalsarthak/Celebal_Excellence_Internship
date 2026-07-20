"""
Streamlit UI: upload a PDF, index it in memory, ask questions about it.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
import faiss

import config
from ingest import extract_pdf_text, chunk_text
from embed_store import embed_texts
from generator import generate_answer

st.set_page_config(page_title="StudyHelp AI", page_icon="📚")
st.title("📚 StudyHelp AI — ask questions from your PDF")

# Session state holds the index + chunks for the currently uploaded file,
# so re-running the script (Streamlit reruns on every interaction) doesn't
# re-embed the same PDF over and over.
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.filename = None

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
    with st.spinner("Reading and indexing your PDF..."):
        # save to a temp path so PdfReader can open it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        text = extract_pdf_text(tmp_path)
        os.remove(tmp_path)

        raw_chunks = chunk_text(text)
        chunks = [
            {"chunk_id": i, "source": uploaded_file.name, "text": c}
            for i, c in enumerate(raw_chunks)
        ]

        embeddings = embed_texts([c["text"] for c in chunks])
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.filename = uploaded_file.name

    st.success(f"Indexed {len(st.session_state.chunks)} chunks from {uploaded_file.name}")

if st.session_state.index is not None:
    question = st.text_input("Ask a question about the PDF")

    if question:
        with st.spinner("Retrieving context and generating answer..."):
            query_embedding = embed_texts([question])
            distances, indices = st.session_state.index.search(query_embedding, config.TOP_K)

            retrieved = [
                {**st.session_state.chunks[idx], "distance": float(dist)}
                for dist, idx in zip(distances[0], indices[0])
                if idx != -1
            ]

            answer = generate_answer(question, retrieved)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Retrieved context (what the answer is grounded in)"):
            for r in retrieved:
                st.markdown(f"**{r['source']}** (distance: {r['distance']:.3f})")
                st.write(r["text"])
                st.divider()
else:
    st.info("Upload a PDF above to get started.")
