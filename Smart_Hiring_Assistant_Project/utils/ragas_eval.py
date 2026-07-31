from datasets import Dataset
from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


def evaluate_rag(question,
                 ground_truth,
                 generated_answer,
                 retrieved_docs):
    """
    Evaluate RAG response using RAGAS.

    Parameters
    ----------
    question : str
    ground_truth : str
    generated_answer : str
    retrieved_docs : list[Document]

    Returns
    -------
    dict
    """

    contexts = [doc.page_content for doc in retrieved_docs]

    dataset = Dataset.from_dict(
        {
            "question": [question],
            "answer": [generated_answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
    )

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return result.to_pandas().iloc[0].to_dict()