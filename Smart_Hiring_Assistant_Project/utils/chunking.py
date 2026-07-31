from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size=1000, chunk_overlap=180):
    """
    Split documents into smaller chunks.

    Args:
        documents: List of LangChain Document objects
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of chunked Document objects
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    return chunks