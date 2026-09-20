"""Local Python tools the ReAct agent can call.

Each tool is a plain Python function that operates on the mock data in
``data.py``. The tool registry at the bottom is what the agent runtime uses
to dispatch a name coming back from the LLM to actual Python code.

Tool descriptions and JSON schemas live here too so ``agent.py`` and
``agent_langgraph.py`` can share the same definitions.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Dict, List

from data import CUSTOMER_NAME_INDEX, CUSTOMERS, TENANT_LOGS, TENANT_METRICS


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def calculate(expression: str) -> Dict[str, Any]:
    """Evaluate a basic arithmetic expression like '125 * 4' or '(10+2)/3'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except Exception as ex:
        return {"expression": expression, "error": f"invalid expression: {ex}"}
    return {"expression": expression, "result": result}


def _resolve_customer_id(customer_id: str) -> str | None:
    if not customer_id:
        return None
    if customer_id in CUSTOMERS:
        return customer_id
    return CUSTOMER_NAME_INDEX.get(customer_id.lower())


def get_customer(customer_id: str) -> Dict[str, Any]:
    """Fetch a customer record (including its tenant_id) by id or name."""
    resolved = _resolve_customer_id(customer_id)
    if not resolved:
        return {"customer_id": customer_id, "error": "customer not found"}
    return CUSTOMERS[resolved]


def get_tenant_metrics(tenant_id: str) -> Dict[str, Any]:
    """Fetch application/database metrics for the tenant."""
    metrics = TENANT_METRICS.get(tenant_id)
    if not metrics:
        return {"tenant_id": tenant_id, "error": "tenant not found"}
    return metrics


def search_logs(tenant_id: str, keyword: str) -> Dict[str, Any]:
    """Return log lines for the tenant that contain the given keyword (case-insensitive)."""
    lines = TENANT_LOGS.get(tenant_id)
    if lines is None:
        return {"tenant_id": tenant_id, "keyword": keyword, "error": "tenant not found"}
    needle = (keyword or "").lower()
    matches: List[str] = [ln for ln in lines if needle in ln.lower()] if needle else list(lines)
    return {
        "tenant_id": tenant_id,
        "keyword": keyword,
        "match_count": len(matches),
        "matches": matches[:20],
    }


# ---------------------------------------------------------------------------
# Registry + schemas
# ---------------------------------------------------------------------------

tool_registry: Dict[str, Callable[..., Any]] = {
    "calculate": calculate,
    "get_customer": get_customer,
    "get_tenant_metrics": get_tenant_metrics,
    "search_logs": search_logs,
}


# JSON schemas in the shape Bedrock Converse expects under toolConfig.tools[].toolSpec.
BEDROCK_TOOL_SPECS = [
    {
        "toolSpec": {
            "name": "calculate",
            "description": (
                "Evaluate a basic arithmetic expression (e.g. '125 * 4', '(10+2)/3'). "
                "Use ONLY for math questions."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Arithmetic expression to evaluate.",
                        }
                    },
                    "required": ["expression"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_customer",
            "description": (
                "Look up a customer by customer id (e.g. 'CUST1001') or by company name "
                "(e.g. 'Acme'). Returns the customer record including its tenant_id."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "Customer id like 'CUST1001', or a company name like 'Acme'.",
                        }
                    },
                    "required": ["customer_id"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_tenant_metrics",
            "description": (
                "Return current application/database metrics for a tenant: active/max DB "
                "connections, CPU%, memory%, error rate%, average query latency, status."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {
                            "type": "string",
                            "description": "Tenant id like 'TENANT-ACME'.",
                        }
                    },
                    "required": ["tenant_id"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "search_logs",
            "description": (
                "Search the local mock log store for lines matching a keyword for a specific "
                "tenant. Useful for finding errors like 'connection', 'timeout', 'exhaustion'."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {
                            "type": "string",
                            "description": "Tenant id like 'TENANT-ACME'.",
                        },
                        "keyword": {
                            "type": "string",
                            "description": "Substring to look for in log lines (case-insensitive).",
                        },
                    },
                    "required": ["tenant_id", "keyword"],
                }
            },
        }
    },
]


def dispatch(tool_name: str, tool_input: Dict[str, Any]) -> Any:
    """Look up ``tool_name`` in the registry and call it with ``tool_input``."""
    if tool_name not in tool_registry:
        return {"error": f"unknown tool: {tool_name}"}
    fn = tool_registry[tool_name]
    try:
        return fn(**(tool_input or {}))
    except TypeError as ex:
        return {"error": f"bad arguments for {tool_name}: {ex}"}
    except Exception as ex:  # noqa: BLE001 - surface runtime errors back to the LLM
        return {"error": f"{tool_name} raised: {ex}"}
