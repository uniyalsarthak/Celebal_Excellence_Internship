import os

from langchain_community.vectorstores import FAISS

FAISS_DB_PATH = os.path.join(
    "vector_store",
    "faiss_index"
)

def create_faiss(chunks, embedding_model):
    """
    Create a FAISS vector database from document chunks
    and save it locally.
    """

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    vector_db.save_local(FAISS_DB_PATH)

    return vector_db


def load_faiss(embedding_model):
    """
    Load an existing FAISS database.
    """

    if not os.path.exists(FAISS_DB_PATH):
        return None

    vector_db = FAISS.load_local(
        folder_path=FAISS_DB_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_db