"""A small tool-using agent.

With ANTHROPIC_API_KEY set: Claude decides which tools to call (tool-use loop).
Without a key: a simple rule-based planner picks the tool, so the demo still runs.
Both paths return the same shape: {"answer": str, "steps": [tool calls], "mode": str}.
"""
import json
import os
import re

from app.tools import TOOL_SPECS, TOOLS

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_TURNS = 6
SYSTEM = (
    "You are a customer-insights agent. Use the provided tools to analyse product "
    "reviews and predict telecom customer churn. Call tools instead of guessing, "
    "then give a short, clear answer."
)


def _call_tool(name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    try:
        return TOOLS[name](**args)
    except Exception as e:  # surface tool errors to the model instead of crashing
        return {"error": str(e)}


def _run_with_claude(query: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": query}]
    steps = []
    for _ in range(MAX_TURNS):
        resp = client.messages.create(
            model=MODEL, max_tokens=1024, system=SYSTEM, tools=TOOL_SPECS, messages=messages
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            return {"answer": text, "steps": steps, "mode": "claude"}
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                out = _call_tool(block.name, block.input)
                steps.append({"tool": block.name, "input": block.input, "output": out})
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out)}
                )
        messages.append({"role": "user", "content": results})
    return {"answer": "Stopped: too many tool turns.", "steps": steps, "mode": "claude"}


def _run_rule_based(query: str) -> dict:
    q = query.lower()
    steps = []
    if "churn" in q:
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", q)]
        contract = next(
            (c for c in ["month-to-month", "one year", "two year"] if c in q),
            "month-to-month" if "month" in q else None,
        )
        if len(nums) >= 2 and contract:
            args = {"tenure_months": int(nums[0]), "contract": contract, "monthly_charges": nums[1]}
            out = _call_tool("predict_churn", args)
            steps.append({"tool": "predict_churn", "input": args, "output": out})
            if "error" not in out:
                answer = (
                    f"Churn probability is {out['churn_probability']:.0%} ({out['risk']} risk). "
                    f"{out['recommendation']}"
                )
                return {"answer": answer, "steps": steps, "mode": "rule-based"}
        return {
            "answer": "For churn, give tenure in months, contract type and monthly charges, "
            "e.g. 'churn for 2 months, month-to-month, 85 monthly'.",
            "steps": steps,
            "mode": "rule-based",
        }
    lines = [l.strip(" -*•") for l in re.split(r"[\n;|]", query) if len(l.strip()) > 3]
    if len(lines) > 1:
        args = {"reviews": lines}
        out = _call_tool("summarize_reviews", args)
        steps.append({"tool": "summarize_reviews", "input": args, "output": out})
        answer = f"Analysed {out['total']} reviews: {out['percentages']}."
        if out["most_negative"]:
            answer += f" Most negative: {out['most_negative']}"
        return {"answer": answer, "steps": steps, "mode": "rule-based"}
    args = {"text": query}
    out = _call_tool("analyze_sentiment", args)
    steps.append({"tool": "analyze_sentiment", "input": args, "output": out})
    return {
        "answer": f"Sentiment: {out['label']} (confidence {out['confidence']}).",
        "steps": steps,
        "mode": "rule-based",
    }


def run_agent(query: str) -> dict:
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return _run_with_claude(query)
        except Exception as e:
            result = _run_rule_based(query)
            result["note"] = f"Claude call failed ({type(e).__name__}); used rule-based fallback."
            return result
    return _run_rule_based(query)
