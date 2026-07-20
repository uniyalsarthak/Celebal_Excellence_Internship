"""
Embedding creation. The vector store itself (FAISS index) is built
directly in app.py per uploaded PDF, kept in memory only.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

import config

_model = None  # lazy-loaded so importing this module doesn't load the model


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype("float32")
