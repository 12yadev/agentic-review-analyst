"""FastAPI service. Run: uvicorn app.api:app --reload  (docs at /docs)"""
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import tools
from app.agent import run_agent

app = FastAPI(title="Agentic Review Analyst", version="1.0.0")


class SentimentIn(BaseModel):
    text: str = Field(min_length=1)


class BatchIn(BaseModel):
    reviews: list[str] = Field(min_length=1)


class ChurnIn(BaseModel):
    tenure_months: int = Field(ge=0, le=100)
    contract: Literal["Month-to-month", "One year", "Two year"]
    monthly_charges: float = Field(gt=0)


class AgentIn(BaseModel):
    query: str = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sentiment")
def sentiment(body: SentimentIn):
    return tools.analyze_sentiment(body.text)


@app.post("/sentiment/batch")
def sentiment_batch(body: BatchIn):
    return tools.summarize_reviews(body.reviews)


@app.post("/churn")
def churn(body: ChurnIn):
    try:
        return tools.predict_churn(body.tenure_months, body.contract, body.monthly_charges)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/agent/run")
def agent_run(body: AgentIn):
    return run_agent(body.query)
