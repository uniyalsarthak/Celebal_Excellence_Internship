import os
import streamlit as st

from utils.pdf_loader import load_pdf
from utils.chunking import split_documents
from utils.embeddings import get_embedding_model
from vector_store.faiss_store import create_faiss
from utils.retriever import get_retriever
from utils.llm import build_rag_chain, ask_question


# Page Configuration
st.set_page_config(
    page_title="Smart Research Assistant",
    layout="wide"
)

st.title(" Smart Research Assistant")
st.markdown("Upload a PDF and ask questions about its contents.")


# Session State
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# PDF Upload
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = ""

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.current_pdf:

        st.session_state.current_pdf = uploaded_file.name
        st.session_state.messages = []
        st.session_state.rag_chain = None
        st.session_state.retriever = None



if uploaded_file:

    # Create folder if it doesn't exist
    os.makedirs("data/uploaded_docs", exist_ok=True)

    pdf_path = os.path.join(
        "data",
        "uploaded_docs",
        uploaded_file.name
    )

    # Save uploaded PDF
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully.")


    # Process PDF
    if st.session_state.rag_chain is None:

        with st.spinner("Processing PDF..."):

            documents = load_pdf(pdf_path)

            chunks = split_documents(documents)

            embedding_model = get_embedding_model()

            vector_db = create_faiss(
                chunks,
                embedding_model
            )

            retriever = get_retriever(vector_db)

            rag_chain = build_rag_chain(retriever)

            st.session_state.retriever = retriever
            st.session_state.rag_chain = rag_chain

        st.success("PDF processed successfully.")



# Chat Interface
if st.session_state.rag_chain:

    st.divider()

    st.subheader(" Chat with your PDF")

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    question = st.chat_input("Ask a question about the uploaded PDF...")

    if question:

        # Store user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Generate response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = ask_question(st.session_state.rag_chain,question)
                # response = st.session_state.rag_chain.invoke(question)

                st.markdown(response)

        # Store assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )





# Sidebar
with st.sidebar:

    st.header("Options")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# Show Source Pages
if (
    st.session_state.rag_chain
    and st.session_state.retriever
    and question
):

    with st.expander("📄 Retrieved Sources"):

        docs = st.session_state.retriever.invoke(question)

        for i, doc in enumerate(docs, start=1):

            page = doc.metadata.get("page", "Unknown")

            source = os.path.basename(
                doc.metadata.get("source", "")
            )

            st.markdown(
                f"""
**Source {i}**

- File : `{source}`
- Page : **{page + 1 if isinstance(page, int) else page}**
"""
            )


# Error Handling
try:
    pass

except Exception as e:
    st.error(f"Error: {e}")