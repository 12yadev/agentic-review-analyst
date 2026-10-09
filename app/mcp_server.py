"""FastMCP server exposing the ML tools to any MCP-compatible agent/client.

Run over stdio:   python -m app.mcp_server
Run over HTTP:    python -m app.mcp_server --http
"""
import sys

from fastmcp import FastMCP

from app import tools

mcp = FastMCP("review-analyst")


@mcp.tool
def analyze_sentiment(text: str) -> dict:
    """Classify a single product review as positive, negative or neutral."""
    return tools.analyze_sentiment(text)


@mcp.tool
def summarize_reviews(reviews: list[str]) -> dict:
    """Classify many reviews; returns counts, percentages and the most negative ones."""
    return tools.summarize_reviews(reviews)


@mcp.tool
def predict_churn(tenure_months: int, contract: str, monthly_charges: float) -> dict:
    """Predict churn probability for a telecom customer.
    contract must be 'Month-to-month', 'One year' or 'Two year'."""
    return tools.predict_churn(tenure_months, contract, monthly_charges)


if __name__ == "__main__":
    if "--http" in sys.argv:
        mcp.run(transport="http", host="127.0.0.1", port=8001)
    else:
        mcp.run()
