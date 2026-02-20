"""Claude RAG response generation using retrieved context."""

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from query.retriever import retrieve

SYSTEM_PROMPT = """You are a Hypertide support agent. Your job is to answer customer support questions using the past support email exchanges provided below as context.

Guidelines:
- Base your answer on the provided context from past support emails
- Match the team's tone and communication style from the examples
- If the context doesn't contain a relevant answer, say so honestly and suggest the customer reach out to support@hypertide.io
- Be concise and helpful
- Do not make up information that isn't supported by the context"""


def build_context_block(hits):
    """Format retrieved Q&A pairs into a context block for the prompt."""
    if not hits:
        return "No relevant past support emails found."

    blocks = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        blocks.append(
            f"--- Past Support Email #{i} ---\n"
            f"Subject: {meta.get('subject', 'N/A')}\n"
            f"Date: {meta.get('date', 'N/A')}\n"
            f"Customer: {meta.get('customer_email', 'N/A')}\n\n"
            f"{hit['document']}\n"
        )

    return "\n".join(blocks)


def generate_response(question, top_k=None):
    """Retrieve context and generate a Claude response for the question."""
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your-anthropic-api-key-here":
        return "Error: ANTHROPIC_API_KEY not set. Add it to your .env file."

    # Retrieve relevant past emails
    hits = retrieve(question, top_k=top_k)
    context = build_context_block(hits)

    # Build the user message
    user_message = (
        f"Here are relevant past support email exchanges:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Customer question: {question}\n\n"
        f"Please provide a helpful response based on the context above."
    )

    # Call Claude
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text
