import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from google import genai
from google.genai import types


def find_project_root(marker: str = "requirements.txt") -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / marker).exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = find_project_root()
DEFAULT_GOLDEN_SET = PROJECT_ROOT / "golden_set.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "final_eval_results.json"

# Load .env from the project root explicitly, regardless of current working directory.
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY or API_KEY.strip() in ("", "your_key_here", "your_gemini_api_key_here"):
    print(
        "GEMINI_API_KEY is missing or still a placeholder.\n"
        f"Expected a '.env' file at: {PROJECT_ROOT / '.env'}\n"
        "with a line: GEMINI_API_KEY=your_real_key"
    )
    sys.exit(1)

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.6-flash"
INTENTS = ["Hardware Issue", "Software Update", "Account & Billing", "General Inquiry"]

MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 5
DEFAULT_SLEEP_BETWEEN_ROWS = float(os.getenv("EVAL_SLEEP_SECONDS", "8"))

HISTORICAL_CONTEXT = "Historical replies: Check settings > general > about. DM us your details."


def _is_rate_limit_error(exc) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "resource_exhausted" in text


def _call_model_with_retry(prompt: str):
    """Returns parsed JSON dict on success. Raises RuntimeError with the real
    reason on failure — never returns a fake fallback value."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
            )
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            last_error = f"Model did not return valid JSON: {e}"
            break  # retrying won't fix a bad prompt/response shape
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            if _is_rate_limit_error(e) and attempt < MAX_RETRIES:
                wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 2)
                print(f"    [rate limited] attempt {attempt}/{MAX_RETRIES}. Waiting {wait:.1f}s...")
                time.sleep(wait)
                continue
            break

    raise RuntimeError(last_error)


def generate_agent_response(query):
    prompt = f"""You are an expert AI support agent for AppleSupport.
User Query: "{query}"
Historical Context: {HISTORICAL_CONTEXT}

Task:
1. Classify the intent into one of: {INTENTS}
2. Decide if this should be escalated to a human. Escalate IF the user is extremely angry or it's physical hardware damage.
3. Draft a helpful, empathetic reply that matches the tone of historical replies.

Output JSON format:
{{
    "intent": "...",
    "escalate": true/false,
    "escalate_reason": "...",
    "draft": "..."
}}
"""
    parsed = _call_model_with_retry(prompt)
    for key in ("intent", "escalate", "escalate_reason", "draft"):
        if key not in parsed:
            raise RuntimeError(f"Agent JSON missing key '{key}'. Got: {parsed}")
    return parsed


def evaluate_with_llm_judge(query, actual_reply, agent_reply):
    prompt = f"""You are an expert evaluator of customer support agents.
User Query: {query}
Actual Human Reply: {actual_reply}
AI Generated Draft: {agent_reply}

Rate the AI Generated Draft on a scale of 1 to 5 based on:
1. Empathy & Tone (Does it sound like Apple Support?)
2. Helpfulness (Does it try to resolve the issue or gather info correctly?)

Output JSON format:
{{
    "score": <int 1-5>,
    "feedback": "..."
}}
"""
    parsed = _call_model_with_retry(prompt)
    for key in ("score", "feedback"):
        if key not in parsed:
            raise RuntimeError(f"Judge JSON missing key '{key}'. Got: {parsed}")
    return parsed["score"], parsed["feedback"]


def preflight_check():
    print("Running preflight check (1 test call)...")
    try:
        result = generate_agent_response("My phone won't turn on after the update.")
        print(f"Preflight OK — sample intent classified as: {result['intent']}")
        return True
    except RuntimeError as e:
        print("\nPreflight FAILED — stopping before the full batch runs.")
        print(f"Reason: {e}")
        print("\nCommon fixes:")
        print(f"  - Confirm a real API key is in: {PROJECT_ROOT / '.env'}")
        print("  - If this was a 429/rate-limit error, wait a bit or check your Gemini quota.")
        return False


def _save_partial(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


def run_evaluation(n_rows: int, sleep_seconds: float, golden_set_path: Path, output_path: Path):
    print(f"Loading Golden Set from {golden_set_path}...")
    if not golden_set_path.exists():
        print(f"ERROR: {golden_set_path} not found. Run prepare_data.py then data_prep.py first.")
        sys.exit(1)

    df = pd.read_csv(golden_set_path)
    if n_rows > 0:
        df = df.head(n_rows)
    total = len(df)
    print(f"Evaluating {total} row(s)...")

    results = []
    scores = []
    failures = 0

    for idx, row in df.iterrows():
        print(f"Evaluating {idx + 1}/{total}...")
        query = row['customer_query']
        actual = row['brand_reply']

        try:
            agent_out = generate_agent_response(query)
        except RuntimeError as e:
            failures += 1
            print(f"  AGENT CALL FAILED: {e}")
            results.append({
                "Query": query,
                "Row_Status": "AGENT_FAILED",
                "Agent_Intent": None,
                "Agent_Escalate": None,
                "Agent_Draft": None,
                "Judge_Score": None,
                "Judge_Feedback": None,
                "Error": str(e),
            })
            _save_partial(results, output_path)
            time.sleep(sleep_seconds)
            continue

        try:
            score, feedback = evaluate_with_llm_judge(query, actual, agent_out['draft'])
            results.append({
                "Query": query,
                "Row_Status": "OK",
                "Agent_Intent": agent_out['intent'],
                "Agent_Escalate": agent_out['escalate'],
                "Agent_Draft": agent_out['draft'],
                "Judge_Score": score,
                "Judge_Feedback": feedback,
                "Error": None,
            })
            scores.append(score)
        except RuntimeError as e:
            failures += 1
            print(f"  JUDGE CALL FAILED: {e}")
            results.append({
                "Query": query,
                "Row_Status": "JUDGE_FAILED",
                "Agent_Intent": agent_out['intent'],
                "Agent_Escalate": agent_out['escalate'],
                "Agent_Draft": agent_out['draft'],
                "Judge_Score": None,
                "Judge_Feedback": None,
                "Error": str(e),
            })

        _save_partial(results, output_path)
        time.sleep(sleep_seconds)

    print("\nEvaluation Complete.")
    print(f"  Rows attempted: {total}")
    print(f"  Rows failed (agent or judge error): {failures}")
    print(f"  Rows successfully scored: {len(scores)}")
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"  Average LLM-Judge Score (successful rows only): {avg_score:.2f}/5.0")
    else:
        print("  No successful rows — cannot compute an average. Fix the failures above and re-run.")

    print(f"  Results written to: {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Rows to evaluate. 0 = all rows.")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_BETWEEN_ROWS)
    parser.add_argument("--golden-set", type=str, default=str(DEFAULT_GOLDEN_SET))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    if args.preflight_only:
        sys.exit(0 if preflight_check() else 1)

    if not args.skip_preflight:
        if not preflight_check():
            sys.exit(1)
        print()

    run_evaluation(
        n_rows=args.n,
        sleep_seconds=args.sleep,
        golden_set_path=Path(args.golden_set),
        output_path=Path(args.output),
    )
