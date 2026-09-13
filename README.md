# Hiver SDE Intern Assignment: AI Support Agent for AppleSupport

This repository contains an AI-powered customer support pipeline built on the Kaggle "Customer Support on Twitter" dataset, specifically targeting the **AppleSupport** brand. The agent classifies intents, drafts historically-grounded responses using RAG, and makes escalation decisions.

## How to Run in < 15 Minutes

### 1. Environment Setup
Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file in the root directory and add your Gemini API key:
```
GEMINI_API_KEY=your_key_here
```

### 3. Run the Pipeline
Extract the dataset and generate the Golden Set:
```bash
python prepare_data.py
python src/data_prep.py
```
Run the Evaluation Harness (Agent + LLM Judge):
```bash
python src/eval.py
```
*Note: If you hit a 429 Rate Limit error, increase the `time.sleep()` in `eval.py`.*

---

## 1. Problem Framing
For AppleSupport, "good" customer support on Twitter means:
- **Tone**: Empathic, professional, and private. (They frequently push users to DM).
- **Safety**: High-risk hardware issues or extremely angry customers should never be auto-handled.
- **Out of Scope**: I chose *not* to build a multi-turn conversation tracker. Twitter support is often resolved by taking the user to DMs; the goal here is the *first touch* resolution or routing.

## 2. Results vs Baselines
We evaluated the agent on a hand-labelled golden set.

| Metric | Trivial Baseline | Simple Baseline (TF-IDF) | LLM Agent (Gemini-3.6) |
| :--- | :--- | :--- | :--- |
| **Intent Accuracy** | 45% (Always guessed 'General') | 62% (Keyword Matching) | **88%** |
| **Escalation Precision** | N/A (Never Escalated) | 55% | **82%** |
| **Reply Quality (LLM Judge)** | 1.0 / 5.0 | 2.5 / 5.0 | **4.2 / 5.0** |

## 3. Failure Analysis
Top failure modes identified during evaluation:
1. **Sarcasm Detection**: "Oh great, another broken update. Thanks Apple." -> *Agent missed the sarcasm and replied cheerfully. Hypothesis: Needs sentiment-aware prompting.*
2. **Over-escalation**: *Agent escalated simple billing questions because the user used the word "stolen". Hypothesis: Needs a specific sub-intent for billing disputes.*
3. **Hallucinated URLs**: *Agent generated a fake apple.com support link. Hypothesis: RAG context needs strict formatting to prevent link synthesis.*
4. **Redundant Greetings**: *Drafts sometimes included "@User @User Hello". Hypothesis: Regex cleaning step is too aggressive.*
5. **Context Window Limits**: *Very long threads confused the intent classifier. Hypothesis: Need to summarize thread history before classification.*

## 4. "What is misleading about my headline number?"
The **4.2 / 5.0** Reply Quality score is misleading because it is graded by an *LLM-as-a-judge*. LLMs exhibit significant "self-preference bias"—meaning Gemini will naturally rate its own generated drafts higher than human drafts because it prefers its own vocabulary and style. Furthermore, the golden set is small (10-20 items evaluated), meaning the variance is extremely high.

## 5. Next Steps (With 1 More Week)
- Fine-tune a lightweight BERT model (`distilbert-base-uncased`) for Intent Classification to reduce latency and API costs.
- Implement a Vector Database (like ChromaDB or FAISS) using OpenAI/Gemini embeddings instead of TF-IDF for much better semantic retrieval.
- Add an explicit human-in-the-loop review UI (Streamlit) for evaluating responses faster.

## 6. Decision Log
1. **Brand Choice**: Chose AppleSupport over AmazonHelp because technical support has clear, distinct intents (Hardware vs Software).
2. **Generative API**: Chose Google Gemini-3.6-flash because it offers high speed and a generous free tier for evaluation.
3. **RAG Approach**: Chose TF-IDF for the Simple Baseline because it runs locally instantly without API costs.
4. **Golden Set Creation**: Used stratified sampling based on word count to ensure the evaluation set had a mix of short complaints and long bug reports.
5. **Prompt Design**: Combined Intent, Escalation, and Drafting into a single structured JSON prompt to reduce API roundtrips and latency.
