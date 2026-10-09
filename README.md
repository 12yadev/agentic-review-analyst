# Agentic Review Analyst

An agentic AI backend built with **Python, FastAPI and FastMCP**. An LLM agent (Claude) decides which
ML tools to call to analyse product-review sentiment and predict telecom customer churn.
Works without an API key too (rule-based planner), so you can demo it anywhere.

## Architecture

```
User query -> POST /agent/run (FastAPI)
                 -> Agent (Claude tool-use loop, or rule-based fallback)
                      -> Tools: analyze_sentiment | summarize_reviews | predict_churn
                           (scikit-learn: TF-IDF + LogisticRegression)
FastMCP server (app/mcp_server.py) exposes the same tools to any MCP client.
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # optional; without it the rule-based agent is used
```

## Run

```bash
uvicorn app.api:app --reload            # API + docs at http://127.0.0.1:8000/docs
python -m app.mcp_server                # MCP server over stdio
python -m app.mcp_server --http         # MCP server over HTTP on :8001
pytest -q                               # tests
```

## Try it

```bash
curl -X POST localhost:8000/sentiment -H "Content-Type: application/json" \
  -d '{"text": "Terrible product, stopped working after two days"}'

curl -X POST localhost:8000/churn -H "Content-Type: application/json" \
  -d '{"tenure_months": 2, "contract": "Month-to-month", "monthly_charges": 85}'

curl -X POST localhost:8000/agent/run -H "Content-Type: application/json" \
  -d '{"query": "What is the churn risk for a customer with 2 months tenure, month-to-month contract, paying 85?"}'
```

Connect the MCP server to Claude Desktop / Claude Code by adding it as a stdio server with command
`python -m app.mcp_server` (run from the project folder).

## Use your own data (recommended before putting this on a resume)

- `SENTIMENT_CSV=path/to/reviews.csv` (columns: `text,label`) - e.g. your product-reviews dataset.
- `CHURN_CSV=path/to/telco.csv` (columns: `tenure,Contract,MonthlyCharges,Churn`) - the 7,043-customer
  Telco dataset from your churn project works directly.

Without these, the app trains on a tiny built-in sample and synthetic churn data (demo only).
