import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from utils import get_llm


# ==========================
# CONFIGURATION
# ==========================

PDF_PATH = r"data\uploaded_docs\VisionPDF_workflow_1.pdf"
OUTPUT_FILE = "test_dataset.json"

# Number of QA pairs to generate from each chunk
QA_PER_CHUNK = 3


# LOAD PDF

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Loaded {len(documents)} pages.")


# PROMPT

prompt = ChatPromptTemplate.from_template(
"""
You are creating a benchmark dataset for evaluating a RAG chatbot.

Generate exactly {num_questions} high-quality question-answer pairs ONLY from the given text.

Rules:

1. Use only the provided text.
2. Do not invent facts.
3. Questions should test understanding.
4. Include factual, conceptual and reasoning questions.
5. Avoid duplicate questions.
6. Return ONLY valid JSON.

Output format:

[
    {{
        "question": "...",
        "ground_truth": "..."
    }}
]

TEXT:

{text}
"""
)

llm = get_llm()

parser = JsonOutputParser()

chain = prompt | llm | parser


# GENERATE DATASET

dataset = []

for i, doc in enumerate(documents):

    print(f"Processing page {i+1}/{len(documents)}")

    try:

        result = chain.invoke(
            {
                "text": doc.page_content,
                "num_questions": QA_PER_CHUNK
            }
        )

        if isinstance(result, list):
            dataset.extend(result)

    except Exception as e:
        print(f"Skipped page {i+1}: {e}")


# REMOVE DUPLICATES

seen = set()
unique_dataset = []

for item in dataset:

    q = item["question"].strip().lower()

    if q not in seen:

        seen.add(q)
        unique_dataset.append(item)


# SAVE

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        unique_dataset,
        f,
        indent=4,
        ensure_ascii=False
    )


print("=" * 60)
print(f"Generated {len(unique_dataset)} QA pairs.")
print(f"Saved to {OUTPUT_FILE}")
print("=" * 60)