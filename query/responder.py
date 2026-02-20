"""Claude RAG response generation using retrieved context."""

import anthropic
from pathlib import Path

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, PROJECT_ROOT
from query.retriever import retrieve

RULES_FILE = PROJECT_ROOT / "rules.txt"

SYSTEM_PROMPT = """You are Hypertide's friendly support assistant on Slack. You answer questions ONLY when you have a clear, confident answer backed by recent email context.

Guidelines:
- Keep responses short and conversational — this is Slack, not email
- Use casual, friendly language (but still professional)
- Get straight to the point — no "Dear customer" or formal greetings
- Use bullet points or short paragraphs, not long blocks of text
- When context emails conflict, ONLY use the most recent one — older emails may be outdated
- Do not make up information that isn't in the context
- NEVER guess or improvise an answer

CRITICAL RULE: If you are not 100% confident that your answer is accurate based on the provided email context, you MUST respond with something like:
"Good question! Let me check on that and get back to you."
Do NOT attempt to answer if there is any doubt. It is better to defer than to give wrong information.

If rules are provided below, they ALWAYS take priority over anything in the email context."""


def load_rules():
    """Load rules from rules.txt, skipping comments and blank lines."""
    if not RULES_FILE.exists():
        return ""
    lines = RULES_FILE.read_text().splitlines()
    rules = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    return "\n".join(rules)


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

    # Load rules
    rules = load_rules()
    rules_block = f"\n\n--- RULES (always follow these) ---\n{rules}\n" if rules else ""

    # Build the user message
    user_message = (
        f"Here are relevant past support email exchanges:\n\n"
        f"{context}\n"
        f"{rules_block}\n"
        f"---\n\n"
        f"Customer question: {question}\n\n"
        f"Please provide a helpful response. Follow the rules above if they apply."
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
