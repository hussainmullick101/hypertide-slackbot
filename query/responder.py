"""Claude RAG response generation using retrieved context."""

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from query.retriever import retrieve

SYSTEM_PROMPT = """You are Hypertide's friendly support assistant on Slack. You answer questions using past support conversations as context.

Guidelines:
- Keep responses short and conversational — this is Slack, not email
- Use casual, friendly language (but still professional)
- Get straight to the point — no "Dear customer" or formal greetings
- Use bullet points or short paragraphs, not long blocks of text
- If you don't have a clear answer from the context, be upfront and suggest they reach out to support@hypertide.io
- Do not make up information that isn't in the context"""


def build_context_block(hits):
    """Format retrieved emails into a context block for the prompt."""
    if not hits:
        return "No relevant past support emails found."

    blocks = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        sender = meta.get("from", meta.get("customer_email", "N/A"))
        blocks.append(
            f"--- Past Support Email #{i} ---\n"
            f"Subject: {meta.get('subject', 'N/A')}\n"
            f"Date: {meta.get('date', 'N/A')}\n"
            f"From: {sender}\n\n"
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
