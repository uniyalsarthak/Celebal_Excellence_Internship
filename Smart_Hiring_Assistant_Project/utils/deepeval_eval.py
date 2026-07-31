from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)


def evaluate_rag(
    question,
    expected_answer,
    generated_answer,
    retrieved_docs,
):
    """
    Evaluate a RAG response using DeepEval.
    """

    contexts = [
        doc.page_content
        for doc in retrieved_docs
    ]

    test_case = LLMTestCase(
        input=question,
        actual_output=generated_answer,
        expected_output=expected_answer,
        retrieval_context=contexts,
    )

    answer_metric = AnswerRelevancyMetric()

    faithfulness_metric = FaithfulnessMetric()

    answer_metric.measure(test_case)
    faithfulness_metric.measure(test_case)

    return {
        "answer_relevancy": answer_metric.score,
        "faithfulness": faithfulness_metric.score,
    }