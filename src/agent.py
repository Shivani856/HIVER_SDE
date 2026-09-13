import os
import json
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors


def find_project_root(marker: str = "requirements.txt") -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / marker).exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = find_project_root()

# Load .env from the project root explicitly, so this works whether you run
# 'python agent.py' from the root or from inside a subfolder like src/.
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY or API_KEY.strip() in ("", "your_key_here", "your_gemini_api_key_here"):
    raise ValueError(
        "GEMINI_API_KEY is missing or still a placeholder.\n"
        f"1. Create a file named '.env' at the project root: {PROJECT_ROOT / '.env'}\n"
        "2. Add a line: GEMINI_API_KEY=your_real_key"
    )

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.6-flash"
INTENTS = ["Hardware Issue", "Software Update", "Account & Billing", "General Inquiry"]

MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 5  # doubles each retry, plus jitter


class AgentCallError(Exception):
    """Raised when the Gemini call fails after all retries. Carries the real reason."""


def _is_rate_limit_error(exc) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "resource_exhausted" in text


def _call_gemini_with_retry(prompt: str) -> str:
    """Calls Gemini with exponential backoff on 429 / transient errors.
    Raises AgentCallError with the real underlying message if every retry fails —
    never swallows the error into a fake success value.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            return response.text
        except (genai_errors.APIError, Exception) as e:  # noqa: BLE001 - we re-raise with context below
            last_error = e
            if _is_rate_limit_error(e) and attempt < MAX_RETRIES:
                wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 2)
                print(f"  [rate limited] attempt {attempt}/{MAX_RETRIES} failed ({e}). "
                      f"Waiting {wait:.1f}s before retry...")
                time.sleep(wait)
                continue
            # Non-rate-limit error, or out of retries: stop immediately.
            break

    raise AgentCallError(
        f"Gemini call failed after {MAX_RETRIES} attempt(s). Last error: {last_error}"
    )


def generate_agent_response(query, historical_replies):
    """Runs intent classification + escalation decision + draft reply in one call.

    Returns a dict with keys: intent, escalate, escalate_reason, draft.
    Raises AgentCallError if the call could not be completed — callers must
    handle this explicitly rather than treating a failure as a valid result.
    """
    context = "\n".join([f"- {r}" for r in historical_replies])

    prompt = f"""You are an expert AI support agent for AppleSupport.

User Query: "{query}"

Historical AppleSupport Replies to similar issues:
{context}

Task:
1. Classify the intent into one of: {INTENTS}
2. Decide if this should be escalated to a human. Escalate IF the user is extremely angry, threatening legal action, or if it's physical hardware damage requiring a store visit.
3. Draft a helpful, empathetic reply that matches the tone of the historical replies.

Output JSON format:
{{
    "intent": "...",
    "escalate": true/false,
    "escalate_reason": "...",
    "draft": "..."
}}
"""
    raw_text = _call_gemini_with_retry(prompt)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise AgentCallError(f"Model did not return valid JSON: {e}. Raw output: {raw_text[:300]!r}")

    for key in ("intent", "escalate", "escalate_reason", "draft"):
        if key not in parsed:
            raise AgentCallError(f"Model JSON is missing required key '{key}'. Got: {parsed}")

    return parsed


if __name__ == "__main__":
    # Simple smoke test — run this file directly to sanity-check your API key/model
    # before running the full evaluation harness.
    q = "My iphone screen cracked when I dropped it, help!"
    hist = ["@User We can help. DM us your model.", "@User Please visit an Apple Store for repair."]
    try:
        result = generate_agent_response(q, hist)
        print("Smoke test OK:")
        print(json.dumps(result, indent=2))
    except AgentCallError as e:
        print(f"Smoke test FAILED: {e}")
