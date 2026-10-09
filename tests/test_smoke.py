import asyncio

from fastapi.testclient import TestClient
from fastmcp import Client

from app.api import app
from app.mcp_server import mcp

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_sentiment():
    r = client.post("/sentiment", json={"text": "Terrible product, stopped working"})
    assert r.status_code == 200
    assert r.json()["label"] == "negative"


def test_churn_risk_ordering():
    hi = client.post(
        "/churn", json={"tenure_months": 2, "contract": "Month-to-month", "monthly_charges": 95}
    ).json()
    lo = client.post(
        "/churn", json={"tenure_months": 60, "contract": "Two year", "monthly_charges": 40}
    ).json()
    assert hi["churn_probability"] > lo["churn_probability"]


def test_churn_validation():
    r = client.post("/churn", json={"tenure_months": 2, "contract": "Weekly", "monthly_charges": 50})
    assert r.status_code == 422


def test_agent_rule_based(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post(
        "/agent/run", json={"query": "churn risk for 2 months, month-to-month, 85 monthly"}
    ).json()
    assert r["steps"][0]["tool"] == "predict_churn"
    assert r["mode"] == "rule-based"


def test_mcp_tools():
    async def go():
        async with Client(mcp) as c:
            names = {t.name for t in await c.list_tools()}
            res = await c.call_tool("analyze_sentiment", {"text": "Great quality, love it"})
            return names, res

    names, res = asyncio.run(go())
    assert names == {"analyze_sentiment", "summarize_reviews", "predict_churn"}
    assert res is not None
