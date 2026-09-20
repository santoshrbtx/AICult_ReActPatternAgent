"""ReAct-style agent using the Bedrock Converse API directly.

This file implements the tool-calling loop by hand, without LangChain /
LangGraph, so the ReAct pattern is easy to follow.

Flow per iteration:

    messages ─► Bedrock.converse(toolConfig=..) ─► assistant message
                                                    │
        if stopReason == "tool_use":                │
            for each toolUse block in the message:  ▼
                execute local Python function
                append tool result as user turn
            loop again
        else:
            print the final assistant text

Nothing about which tool to call is hardcoded here; the LLM decides.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from bedrock_client import (
    MAX_TOKENS,
    get_model_id,
    invoke_with_refresh,
    make_client,
)
from tools import BEDROCK_TOOL_SPECS, dispatch

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

MAX_ITERATIONS = 8


# ---------------------------------------------------------------------------
# Rich printing helpers
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


# ---------------------------------------------------------------------------
# Converse response helpers
# ---------------------------------------------------------------------------

def _extract_assistant_text(assistant_message: Dict[str, Any]) -> str:
    parts = []
    for block in assistant_message.get("content", []):
        if "text" in block and block["text"]:
            parts.append(block["text"])
    return "\n".join(parts).strip()


def _extract_tool_uses(assistant_message: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [block["toolUse"] for block in assistant_message.get("content", []) if "toolUse" in block]


# ---------------------------------------------------------------------------
# "What is being sent to the LLM" printer
# ---------------------------------------------------------------------------

def _print_llm_input(
    iteration: int,
    model_id: str,
    system_prompt: str,
    tool_specs: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    inference_config: Dict[str, Any],
    show_full_tools: bool,
) -> None:
    """Print exactly what is about to be POSTed to Bedrock Converse.

    On iteration 1 the full tool JSON schemas are shown; on later iterations
    only the tool name+description summary is repeated (they don't change),
    so the console stays readable.
    """
    _print_rule(f"ITERATION {iteration} — LLM INPUT (what we send to Bedrock)")

    console.print(f"[bold]modelId:[/bold] [magenta]{model_id}[/magenta]")

    console.print("[bold]system prompt:[/bold]")
    console.print(Panel(system_prompt, border_style="green", expand=True))

    if show_full_tools:
        console.print(f"[bold]toolConfig.tools ({len(tool_specs)} tools):[/bold]")
        console.print(
            Syntax(
                json.dumps({"tools": tool_specs}, indent=2, default=str),
                "json",
                theme="monokai",
                word_wrap=True,
            )
        )
    else:
        tool_table = Table(title=f"toolConfig.tools ({len(tool_specs)} tools — schemas unchanged)",
                           show_header=True, header_style="bold cyan")
        tool_table.add_column("name", style="green")
        tool_table.add_column("description")
        for t in tool_specs:
            spec = t["toolSpec"]
            tool_table.add_row(spec["name"], spec["description"])
        console.print(tool_table)

    console.print(f"[bold]messages ({len(messages)} turn(s)):[/bold]")
    console.print(
        Syntax(
            json.dumps(messages, indent=2, default=str),
            "json",
            theme="monokai",
            word_wrap=True,
        )
    )

    console.print("[bold]inferenceConfig:[/bold]")
    console.print(
        Syntax(
            json.dumps(inference_config, indent=2, default=str),
            "json",
            theme="monokai",
            word_wrap=True,
        )
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_agent(user_question: str) -> str:
    client = make_client()
    model_id = get_model_id()

    _print_header("USER QUESTION")
    console.print(f"[bold white]{user_question}[/bold white]")

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"text": user_question}]}
    ]

    inference_config: Dict[str, Any] = {"maxTokens": MAX_TOKENS}

    final_text = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        _print_llm_input(
            iteration=iteration,
            model_id=model_id,
            system_prompt=SYSTEM_PROMPT,
            tool_specs=BEDROCK_TOOL_SPECS,
            messages=messages,
            inference_config=inference_config,
            show_full_tools=(iteration == 1),
        )

        response = invoke_with_refresh(
            client.converse,
            modelId=model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig={"tools": BEDROCK_TOOL_SPECS},
            inferenceConfig=inference_config,
        )

        stop_reason = response.get("stopReason")
        assistant_message = response["output"]["message"]
        messages.append(assistant_message)

        assistant_text = _extract_assistant_text(assistant_message)
        tool_uses = _extract_tool_uses(assistant_message)

        _print_rule(f"ITERATION {iteration} — LLM RESPONSE / TOOL SELECTION")

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="bold green")
        table.add_column()
        table.add_row("stopReason", f"[magenta]{stop_reason}[/magenta]")
        table.add_row("tool_calls", f"[cyan]{[tu['name'] for tu in tool_uses] or '<none>'}[/cyan]")
        console.print(table)

        if assistant_text:
            console.print(f"[bold]assistant_text:[/bold]\n{assistant_text}")

        if stop_reason != "tool_use" or not tool_uses:
            final_text = assistant_text
            break

        # Execute every tool call the LLM requested this turn.
        tool_result_blocks: List[Dict[str, Any]] = []
        for tu in tool_uses:
            tool_name = tu["name"]
            tool_input = tu.get("input", {}) or {}
            tool_use_id = tu["toolUseId"]

            _print_rule(f"TOOL CALL — {tool_name}")
            _print_kv("TOOL NAME", tool_name)
            _print_kv("TOOL ARGUMENTS", tool_input)

            result = dispatch(tool_name, tool_input)

            _print_kv("TOOL RESULT", result)

            tool_result_blocks.append({
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"json": result if isinstance(result, dict) else {"value": result}}],
                }
            })

        # Feed all tool results back as a single user turn.
        messages.append({"role": "user", "content": tool_result_blocks})

    else:
        _print_rule("MAX ITERATIONS REACHED")
        console.print(f"[red]Stopped after {MAX_ITERATIONS} iterations without a final answer.[/red]")

    _print_header("FINAL ANSWER")
    console.print(Markdown(final_text or "*No final answer produced.*"))
    return final_text


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

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
