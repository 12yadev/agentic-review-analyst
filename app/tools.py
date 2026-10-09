"""Core ML tools shared by the MCP server, the FastAPI app and the agent.

Sentiment: TF-IDF + Logistic Regression (trained on a small built-in sample;
swap in your own reviews CSV via SENTIMENT_CSV with columns: text,label).
Churn: Logistic Regression on tenure / contract / monthly charges
(demo data is synthetic; swap in the Telco dataset via CHURN_CSV with columns:
tenure, Contract, MonthlyCharges, Churn).
"""
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

POS = [
    "Absolutely love this product, works perfectly",
    "Great quality and fast delivery, very happy",
    "Excellent value for money, highly recommend",
    "Amazing battery life and the screen is gorgeous",
    "Superb build quality, exceeded my expectations",
    "Very satisfied, easy to use and reliable",
    "Fantastic purchase, will buy again",
    "The sound quality is wonderful and setup was simple",
    "Best phone I have owned, smooth and fast",
    "Really good product, totally worth the price",
    "Customer support was helpful and the item is perfect",
    "Brilliant design and it feels premium",
]
NEG = [
    "Terrible product, stopped working after two days",
    "Very disappointed, poor quality and bad packaging",
    "Waste of money, do not buy this",
    "Battery drains quickly and the screen is dull",
    "Awful experience, support never replied",
    "The item arrived broken and refund was slow",
    "Worst purchase ever, completely useless",
    "Cheap material and it feels flimsy",
    "Very slow and keeps crashing, horrible",
    "Not as described, I regret buying it",
    "Overpriced and unreliable, extremely unhappy",
    "Sound is awful and it stopped charging",
]
NEU = [
    "It is okay, does the job",
    "Average product, nothing special",
    "Delivery was on time, product is as expected",
    "Decent for the price, not great not bad",
    "It works, but there is nothing remarkable about it",
    "Fine for basic use",
    "The product is acceptable, a bit ordinary",
    "Neither good nor bad, just average",
    "Received the item, it functions normally",
    "Mediocre but usable",
    "Standard quality, as shown in the pictures",
    "So-so experience overall",
]


@lru_cache(maxsize=1)
def _sentiment_model() -> Pipeline:
    path = os.getenv("SENTIMENT_CSV")
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        X, y = df["text"].astype(str), df["label"].astype(str)
    else:
        X = POS + NEG + NEU
        y = ["positive"] * len(POS) + ["negative"] * len(NEG) + ["neutral"] * len(NEU)
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, C=5.0)),
        ]
    )
    model.fit(X, y)
    return model


CONTRACTS = ["Month-to-month", "One year", "Two year"]


def _synthetic_churn(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20])
    charges = rng.uniform(20, 120, n).round(2)
    z = (
        -0.8
        - 0.045 * tenure
        + 0.018 * charges
        + np.where(contract == "Month-to-month", 1.6, np.where(contract == "One year", 0.3, -0.8))
    )
    churn = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame(
        {"tenure": tenure, "Contract": contract, "MonthlyCharges": charges, "Churn": churn}
    )


@lru_cache(maxsize=1)
def _churn_model() -> Pipeline:
    path = os.getenv("CHURN_CSV")
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        df["Churn"] = df["Churn"].map(lambda v: 1 if str(v).lower() in ("1", "yes", "true") else 0)
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        df = df.dropna(subset=["tenure", "Contract", "MonthlyCharges"])
    else:
        df = _synthetic_churn()
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), ["tenure", "MonthlyCharges"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Contract"]),
        ]
    )
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(df[["tenure", "MonthlyCharges", "Contract"]], df["Churn"])
    return model


# ----------------------------- tool functions -----------------------------

def analyze_sentiment(text: str) -> dict:
    """Classify one review as positive, negative or neutral."""
    model = _sentiment_model()
    probs = model.predict_proba([text])[0]
    idx = int(np.argmax(probs))
    return {
        "text": text,
        "label": str(model.classes_[idx]),
        "confidence": round(float(probs[idx]), 3),
    }


def summarize_reviews(reviews: list[str]) -> dict:
    """Classify many reviews and return counts, percentages and the most negative ones."""
    if not reviews:
        return {"total": 0, "counts": {}, "percentages": {}, "most_negative": []}
    results = [analyze_sentiment(r) for r in reviews]
    counts: dict[str, int] = {}
    for r in results:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    total = len(results)
    negs = sorted((r for r in results if r["label"] == "negative"), key=lambda r: -r["confidence"])
    return {
        "total": total,
        "counts": counts,
        "percentages": {k: round(100 * v / total, 1) for k, v in counts.items()},
        "most_negative": [r["text"] for r in negs[:3]],
    }


def predict_churn(tenure_months: int, contract: str, monthly_charges: float) -> dict:
    """Estimate churn probability for a telecom customer."""
    contract_norm = next((c for c in CONTRACTS if c.lower() == contract.strip().lower()), None)
    if contract_norm is None:
        raise ValueError(f"contract must be one of {CONTRACTS}")
    row = pd.DataFrame(
        [{"tenure": tenure_months, "MonthlyCharges": monthly_charges, "Contract": contract_norm}]
    )
    p = float(_churn_model().predict_proba(row)[0][1])
    risk = "high" if p >= 0.6 else "medium" if p >= 0.35 else "low"
    tip = {
        "high": "Offer a discounted 1-2 year contract and a proactive retention call.",
        "medium": "Send a loyalty offer and check recent support tickets.",
        "low": "No action needed; keep monitoring.",
    }[risk]
    return {
        "churn_probability": round(p, 3),
        "risk": risk,
        "contract": contract_norm,
        "recommendation": tip,
    }


TOOLS = {
    "analyze_sentiment": analyze_sentiment,
    "summarize_reviews": summarize_reviews,
    "predict_churn": predict_churn,
}

# JSON schemas in Anthropic tool-use format
TOOL_SPECS = [
    {
        "name": "analyze_sentiment",
        "description": "Classify a single product review as positive, negative or neutral.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The review text"}},
            "required": ["text"],
        },
    },
    {
        "name": "summarize_reviews",
        "description": "Classify a list of reviews and return sentiment counts, percentages and the most negative reviews.",
        "input_schema": {
            "type": "object",
            "properties": {"reviews": {"type": "array", "items": {"type": "string"}}},
            "required": ["reviews"],
        },
    },
    {
        "name": "predict_churn",
        "description": "Predict churn probability for a telecom customer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenure_months": {"type": "integer"},
                "contract": {"type": "string", "enum": CONTRACTS},
                "monthly_charges": {"type": "number"},
            },
            "required": ["tenure_months", "contract", "monthly_charges"],
        },
    },
]
