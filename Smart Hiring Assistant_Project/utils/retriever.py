def get_retriever(vector_db):
    """
    Create and return a retriever using
    Max Marginal Relevance (MMR).
    """

    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,         # Final chunks sent to LLM
            "fetch_k": 15   # Initial candidates considered
        }
    )

    return retriever