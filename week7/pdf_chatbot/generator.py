"""
Stage 7: Answer Generation

Uses the Groq API to generate an answer
grounded in the retrieved context. Groq offers a free tier for
developers, and its inference is very fast.
"""

import requests
import config


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    return f"""You are answering a question using ONLY the context below.
If the answer isn't contained in the context, say you don't have enough
information rather than guessing.

Context:
{context_block}

Question: {question}

Answer:"""


def generate_answer(question: str, chunks: list[dict]) -> str:
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "Set the GROQ_API_KEY environment variable (in .env) before calling generate_answer()."
        )

    prompt = build_prompt(question, chunks)

    response = requests.post(
        f"{config.GROQ_API_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
        },
        timeout=60,
    )
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]
