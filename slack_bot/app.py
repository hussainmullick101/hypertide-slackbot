"""Slack Bolt bot for the Hypertide support knowledge base — DMs only."""

import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from config import SLACK_BOT_TOKEN, SLACK_APP_TOKEN, ANTHROPIC_API_KEY, CLAUDE_MODEL
from query.responder import generate_response, build_context_block
from query.retriever import retrieve

import anthropic

app = App(token=SLACK_BOT_TOKEN)

NO_RESPONSE_NEEDED = "NO_RESPONSE_NEEDED"

TRIAGE_SYSTEM_PROMPT = """You are a triage filter. Decide if this Slack DM needs a support response.
Reply with exactly NO_RESPONSE_NEEDED (and nothing else) if the message is:
- A simple "thanks", "thank you", "got it", "ok", "appreciate it", or similar
- A greeting with no question ("hi", "hello", "hey")
- An emoji-only message
- Any message that doesn't contain a question or request

Otherwise, reply with exactly RESPONSE_NEEDED (and nothing else)."""


def needs_response(text: str) -> bool:
    """Return True if the message warrants a RAG-powered response."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=20,
        system=TRIAGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    return NO_RESPONSE_NEEDED not in resp.content[0].text


@app.event("message")
def handle_dm(event, say):
    """Respond to direct messages with a RAG-powered answer."""
    # Skip bot messages and thread replies to avoid loops
    if event.get("bot_id") or event.get("subtype"):
        return

    question = event.get("text", "").strip()
    if not question:
        return

    # Filter out messages that don't need a response
    if not needs_response(question):
        print(f"Skipping (no response needed): {question!r}")
        return

    try:
        answer = generate_response(question)
        say(text=answer)
    except Exception as e:
        say(text=f"Sorry, I ran into an error: {e}")


def main():
    if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
        print("Error: SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set in .env")
        print("Create a Slack app at https://api.slack.com/apps first.")
        return

    print("Starting Hypertide Slack Support Bot (DMs only)...")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()


if __name__ == "__main__":
    main()
