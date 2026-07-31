import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.output_parsers import StrOutputParser

from utils.prompt import get_prompt

load_dotenv()


def get_llm():
    """
    Initialize the Groq LLM.
    """

    return ChatGroq(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )


def build_rag_chain(retriever):
    """
    Build the RAG pipeline.
    """

    llm = get_llm()

    prompt = get_prompt()

    parser = StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    chain = (
        {
            "context": retriever | format_docs,
            "question": lambda question: question
        }
        | prompt
        | llm
        | parser
    )

    return chain


def ask_question(rag_chain, question):
    """
    Generate an answer using the RAG chain.
    """

    return rag_chain.invoke(question)