"""ReAct-style agent using LangGraph's prebuilt ``create_react_agent``.

Same tools, same data, same behaviour as ``agent.py`` — but the tool-calling
loop is handled by the LangGraph runtime instead of being hand-written.

Compare this file with ``agent.py`` to see what the framework hides from you.
"""

from __future__ import annotations

import sys
import json
from typing import Any, Dict, List

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from bedrock_client import MAX_TOKENS, get_model_id, get_region
from tools import (
    calculate as _calculate,
    get_customer as _get_customer,
    get_tenant_metrics as _get_tenant_metrics,
    search_logs as _search_logs,
)

console = Console()

SYSTEM_PROMPT = (
    "You are a helpful support engineer assistant. "
    "You have access to local tools that can look up customers, fetch tenant metrics, "
    "search application logs, and perform arithmetic. "
    "When a question requires data, call the appropriate tool. "
    "Chain tools together when needed (e.g. get the customer first to find their tenant_id, "
    "then fetch metrics/logs). "
    "When you have enough information, produce a concise final answer grounded in the tool results. "
    "If a question does not need any tool, just answer directly."
)


# ---------------------------------------------------------------------------
# LangChain-flavoured tool wrappers (they just delegate to the plain functions)
# ---------------------------------------------------------------------------

@tool
def calculate(expression: str) -> Dict[str, Any]:
    """Evaluate a basic arithmetic expression like '125 * 4' or '(10+2)/3'."""
    return _calculate(expression)


@tool
def get_customer(customer_id: str) -> Dict[str, Any]:
    """Look up a customer by id (e.g. 'CUST1001') or company name (e.g. 'Acme').

    Returns the customer record including its tenant_id.
    """
    return _get_customer(customer_id)


@tool
def get_tenant_metrics(tenant_id: str) -> Dict[str, Any]:
    """Return current app/db metrics for a tenant: connections, CPU, memory, error rate."""
    return _get_tenant_metrics(tenant_id)


@tool
def search_logs(tenant_id: str, keyword: str) -> Dict[str, Any]:
    """Search the local mock log store for lines matching a keyword for a tenant."""
    return _search_logs(tenant_id, keyword)


LC_TOOLS = [calculate, get_customer, get_tenant_metrics, search_logs]


# ---------------------------------------------------------------------------
# Rich printing helpers (match agent.py look-and-feel)
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    console.print()
    console.print(Panel(f"[bold cyan]{title}[/bold cyan]", expand=True, border_style="cyan"))


def _print_rule(title: str) -> None:
    console.print()
    console.rule(f"[bold yellow]{title}[/bold yellow]", style="yellow")


def _print_kv(label: str, value: Any) -> None:
    if isinstance(value, (dict, list)):
        rendered = json.dumps(value, indent=2, default=str)
        console.print(f"[bold]{label}:[/bold]")
        console.print(Syntax(rendered, "json", theme="monokai", word_wrap=True))
    else:
        console.print(f"[bold]{label}:[/bold] {value}")


def _print_json(obj: Any) -> None:
    console.print(
        Syntax(json.dumps(obj, indent=2, default=str), "json", theme="monokai", word_wrap=True)
    )


# ---------------------------------------------------------------------------
# Agent construction + run
# ---------------------------------------------------------------------------

def _build_agent():
    llm = ChatBedrockConverse(
        model=get_model_id(),
        region_name=get_region(),
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    return create_react_agent(llm, tools=LC_TOOLS)


def _stringify_tool_content(content: Any) -> str:
    if isinstance(content, (dict, list)):
        return json.dumps(content, indent=2, default=str)
    return str(content)


def _print_llm_input(user_question: str) -> None:
    """Print exactly what LangGraph will hand to Bedrock on the first LLM call.

    LangGraph re-derives Bedrock's toolConfig from the @tool-decorated
    functions and grows the messages list itself, so we show the initial
    setup once (system prompt, tool inventory, first user message,
    inference config) rather than per iteration.
    """
    _print_rule("LLM INPUT (what LangGraph hands to Bedrock)")
    console.print(f"[bold]modelId:[/bold] [magenta]{get_model_id()}[/magenta]")
    console.print(f"[bold]region:[/bold]  [magenta]{get_region()}[/magenta]")

    console.print("[bold]system prompt:[/bold]")
    console.print(Panel(SYSTEM_PROMPT, border_style="green", expand=True))

    tool_table = Table(
        title=f"tools ({len(LC_TOOLS)}) — derived from @tool-decorated functions",
        show_header=True,
        header_style="bold cyan",
    )
    tool_table.add_column("name", style="green")
    tool_table.add_column("description")
    tool_table.add_column("args", style="dim")
    for t in LC_TOOLS:
        try:
            args_schema = t.args
        except Exception:  # noqa: BLE001
            args_schema = {}
        tool_table.add_row(t.name, t.description, json.dumps(args_schema, default=str))
    console.print(tool_table)

    console.print("[bold]initial messages:[/bold]")
    _print_json([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ])

    console.print("[bold]inferenceConfig (set on ChatBedrockConverse):[/bold]")
    _print_json({"max_tokens": MAX_TOKENS, "temperature": 0.0})


# ---------------------------------------------------------------------------
# End-of-run comparison ("quick byte")
# ---------------------------------------------------------------------------

def _print_langgraph_vs_plain_comparison() -> None:
    _print_rule("LANGGRAPH vs PLAIN AGENT — QUICK BYTE")

    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Aspect", style="bold green", no_wrap=True)
    table.add_column("agent.py (no framework)")
    table.add_column("agent_langgraph.py (this file)")

    table.add_row(
        "Loop control",
        "Explicit for-loop; you read stopReason, dispatch tools, append toolResult.",
        "Hidden inside create_react_agent — you just call agent.stream(inputs).",
    )
    table.add_row(
        "Tool schemas",
        "Hand-written JSON Schema in BEDROCK_TOOL_SPECS.",
        "Auto-derived from @tool-decorated functions (name, docstring, type hints).",
    )
    table.add_row(
        "Messages format",
        "Raw Bedrock dicts with toolUse / toolResult blocks.",
        "Typed classes: SystemMessage / HumanMessage / AIMessage / ToolMessage.",
    )
    table.add_row(
        "State model",
        "A plain Python list you append to yourself.",
        "A LangGraph state graph (AgentState) checkpointable & streamable.",
    )
    table.add_row(
        "Observability",
        "Whatever you print. Full control, zero magic.",
        "agent.stream() emits per-node updates; also plugs into LangSmith.",
    )
    table.add_row(
        "Extensibility",
        "Add features by editing the loop.",
        "Add nodes/edges (memory, retries, human-in-the-loop) with graph APIs.",
    )
    table.add_row(
        "Dependencies",
        "boto3 only.",
        "boto3 + langchain-core + langchain-aws + langgraph.",
    )
    table.add_row(
        "Best for",
        "Learning the ReAct pattern; small, fully-transparent agents.",
        "Production agents where you want retries, memory, branching, tracing.",
    )
    console.print(table)

    console.print(
        Panel(
            "[bold]TL;DR[/bold] — Both send the SAME 4 inputs to Bedrock "
            "(system prompt, tool config, messages, inference config) and get the "
            "SAME wire-format toolUse/toolResult blocks back. LangGraph just packages "
            "that loop as a reusable state graph so you stop writing it by hand.",
            border_style="magenta",
            expand=True,
        )
    )


def run_agent(user_question: str) -> str:
    _print_header("USER QUESTION")
    console.print(f"[bold white]{user_question}[/bold white]")

    _print_llm_input(user_question)

    agent = _build_agent()

    inputs = {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_question)]}

    step = 0
    final_text = ""
    seen_message_ids: set[str] = set()

    for update in agent.stream(inputs, stream_mode="values"):
        messages: List = update.get("messages", [])
        # Print only newly-added messages this tick.
        for msg in messages:
            msg_id = getattr(msg, "id", None) or f"{type(msg).__name__}:{id(msg)}"
            if msg_id in seen_message_ids:
                continue
            seen_message_ids.add(msg_id)

            if isinstance(msg, AIMessage):
                step += 1
                _print_rule(f"ITERATION {step} — LLM RESPONSE / TOOL SELECTION")
                tool_calls = getattr(msg, "tool_calls", None) or []

                text = msg.content if isinstance(msg.content, str) else ""
                if not text and isinstance(msg.content, list):
                    # Bedrock via LangChain sometimes returns list-of-blocks content.
                    text = "\n".join(
                        block.get("text", "") for block in msg.content if isinstance(block, dict)
                    ).strip()

                summary = Table(show_header=False, box=None, padding=(0, 1))
                summary.add_column(style="bold green")
                summary.add_column()
                summary.add_row(
                    "tool_calls",
                    f"[cyan]{[tc['name'] for tc in tool_calls] or '<none>'}[/cyan]",
                )
                console.print(summary)

                if text:
                    console.print(f"[bold]assistant_text:[/bold]\n{text}")
                if not tool_calls and text:
                    final_text = text
            elif isinstance(msg, ToolMessage):
                _print_rule(f"TOOL RESULT — {msg.name}")
                _print_kv("TOOL NAME", msg.name)
                _print_kv("TOOL RESULT", _stringify_tool_content(msg.content))
            # HumanMessage / SystemMessage: nothing to print beyond the header we already showed.

    _print_header("FINAL ANSWER")
    console.print(Markdown(final_text or "*No final answer produced.*"))

    _print_langgraph_vs_plain_comparison()
    return final_text


DEMO_QUESTIONS = [
    "What is 125 * 4?",
    "Tell me about customer CUST1001.",
    "Why is the Acme tenant experiencing database connection problems?",
    "In one sentence, what does the acronym 'SQL' stand for?",
]


def main() -> None:
    if len(sys.argv) > 1:
        run_agent(" ".join(sys.argv[1:]))
        return

    for i, question in enumerate(DEMO_QUESTIONS, 1):
        _print_header(f"DEMO {i}/{len(DEMO_QUESTIONS)}")
        run_agent(question)


if __name__ == "__main__":
    main()
