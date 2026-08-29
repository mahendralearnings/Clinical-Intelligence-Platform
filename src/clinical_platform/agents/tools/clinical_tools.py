"""
Tools the agent can choose to call. Each tool is a plain Python function
decorated with @tool - LangChain reads the function's docstring and type
hints to tell the LLM what the tool does and what input it expects.

Design note: search_clinical_documents wraps our EXISTING RetrievalService
from Phase 3 - we are NOT rebuilding search, just exposing the existing,
already-tested search as something the agent can decide to call.
"""

import ast
import operator

from langchain_core.tools import tool

from clinical_platform.services.retrieval_service import RetrievalService

# Populated by whoever builds the graph (see clinical_agent_graph.py).
# A simple module-level reference - the tool functions themselves must be
# plain functions (LangChain's @tool decorator requires this), so we can't
# pass RetrievalService in through a constructor like our other services.
_retrieval_service: RetrievalService | None = None


def set_retrieval_service(service: RetrievalService) -> None:
    global _retrieval_service
    _retrieval_service = service


@tool
def search_clinical_documents(query: str) -> str:
    """
    Search the clinical document library (SOPs, drug manuals, trial
    summaries) for information relevant to the query. Returns the top
    matching excerpts with their source document and section.
    """
    if _retrieval_service is None:
        return "Error: search service not configured."

    results = _retrieval_service.search(query, top_k=3, min_score=0.5)
    if not results:
        return "No relevant documents found for this query."

    lines = []
    for chunk, score in results:
        lines.append(
            f"[Source: {chunk.metadata.source} | "
            f"Section: {chunk.metadata.section_title}]\n{chunk.content}"
        )
    return "\n\n".join(lines)


# Only a safe, restricted set of arithmetic operations - never use Python's
# real eval() on user/LLM-provided input, that's a code execution risk.
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    raise ValueError("Only basic arithmetic (+, -, *, /) is allowed.")


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a simple arithmetic expression, e.g. "2550 / 3" or "500 * 2".
    Only basic addition, subtraction, multiplication, and division are
    supported. Use this for dosage math or other simple calculations.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as exc:
        return f"Could not calculate '{expression}': {exc}"