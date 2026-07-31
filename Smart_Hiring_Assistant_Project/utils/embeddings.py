import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings


@st.cache_resource
def get_embedding_model():
    """
    Load the embedding model only once.
    """

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )