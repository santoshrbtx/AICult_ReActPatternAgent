# AICult_ReActPatternAgent

A small, educational **Local ReAct Agent POC** powered by **Amazon Bedrock**.
It ships **two** implementations of the same agent so you can compare them
side by side:

1. `agent.py` — hand-written ReAct/tool-calling loop using the Bedrock
   Converse API directly. **No LangChain, no LangGraph.**
2. `agent_langgraph.py` — same tools & data, but the loop is delegated to
   LangGraph's prebuilt `create_react_agent`.

The point of the POC is to show the fundamental relationship between the
LLM, tool selection, the agent runtime, and tool execution:

```text
User Question
     ↓
Bedrock LLM
     ↓
Select appropriate tool
     ↓
Execute local Python tool
     ↓
Return tool result to LLM
     ↓
LLM decides next action
     ↓
Repeat until task is complete
     ↓
Final Answer
```

The LLM **chooses** the tool. The agent runtime **runs** the Python function.

## Project layout

```text
AICult_ReActPatternAgent/
├── .venv/                  # created by you
├── bedrock_client.py       # shared Bedrock client factory (region, model, SSO refresh)
├── agent.py                # ReAct loop WITHOUT LangGraph
├── agent_langgraph.py      # ReAct loop WITH LangGraph
├── tools.py                # 4 local tools + tool registry + Bedrock tool schemas
├── data.py                 # mock customers / tenant metrics / tenant logs
├── requirements.txt
└── README.md
```

## The 4 tools

| Tool                 | Signature                                    | What it does                                     |
| -------------------- | -------------------------------------------- | ------------------------------------------------ |
| `calculate`          | `calculate(expression)`                      | Safe arithmetic evaluator                        |
| `get_customer`       | `get_customer(customer_id)`                  | Looks up customer + tenant id (id or name)       |
| `get_tenant_metrics` | `get_tenant_metrics(tenant_id)`              | Returns mock DB/app metrics for the tenant       |
| `search_logs`        | `search_logs(tenant_id, keyword)`            | Grep-like search over a mock in-memory log store |

All data lives in `data.py` — no CloudWatch, DynamoDB, S3, Lambda, or vector
DBs are used. Bedrock is the only AWS service touched.

## Setup

### 1. Create a virtual environment

Windows (PowerShell / bash-on-Windows):

```bash
python -m venv .venv
source .venv/Scripts/activate   # bash
# or: .venv\Scripts\Activate.ps1  # PowerShell
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

Minimum (needed by `agent.py`):

```bash
pip install boto3 botocore
```

Full install (also enables `agent_langgraph.py`):

```bash
pip install -r requirements.txt
```

### 3. Configure Bedrock access

`bedrock_client.py` is the single source of truth for Bedrock config. It
reads the following environment variables (all optional):

| Var                          | Default                          |
| ---------------------------- | -------------------------------- |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | `us-east-1`                |
| `AWS_PROFILE`                | *(uses default profile)*         |
| `BEDROCK_MODEL_ID`           | `us.anthropic.claude-opus-5`     |
| `BEDROCK_EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v2:0`   |

Log in via AWS SSO before running (the client will auto-attempt a refresh
via `aws sso login` if the token has expired):

```bash
aws sso login            # or: aws sso login --profile <your-profile>
```

## Run it

### Option A — plain Python ReAct (recommended first)

Run the built-in demo scenarios (math → single-tool → multi-tool → no-tool):

```bash
python agent.py
```

Or ask a specific question:

```bash
python agent.py "Why is the Acme tenant experiencing database connection problems?"
python agent.py "What is 125 * 4?"
python agent.py "Tell me about customer CUST1001."
python agent.py "In one sentence, what is a REST API?"
```

### Option B — LangGraph version

```bash
python agent_langgraph.py
python agent_langgraph.py "Why is the Acme tenant experiencing database connection problems?"
```

Both produce the same shape of console output so the ReAct pattern is
easy to observe:

```text
=========================
USER QUESTION
=========================
Why is the Acme tenant experiencing database connection problems?

-------------------------
ITERATION 1 — LLM RESPONSE / TOOL SELECTION
-------------------------
stopReason: tool_use
tool_calls: ['get_customer']

-------------------------
TOOL CALL — get_customer
-------------------------
TOOL NAME: get_customer
TOOL ARGUMENTS: { "customer_id": "Acme" }
TOOL RESULT:  { "customer_id": "CUST1001", "tenant_id": "TENANT-ACME", ... }

-------------------------
ITERATION 2 — LLM RESPONSE / TOOL SELECTION
-------------------------
stopReason: tool_use
tool_calls: ['get_tenant_metrics']
...

=========================
FINAL ANSWER
=========================
Acme (TENANT-ACME) is at 198/200 DB connections; logs show connection
pool exhaustion + acquisition timeouts + long-running queries...
```

## What is actually sent to the LLM

Every ReAct iteration calls Bedrock's Converse API with the same four
pieces of input. Understanding these is the whole POC — everything else
is plumbing.

```python
client.converse(
    modelId       = "us.anthropic.claude-opus-5",
    system        = [{"text": SYSTEM_PROMPT}],   # (1) system prompt
    toolConfig    = {"tools": BEDROCK_TOOL_SPECS},# (2) tool definitions
    messages      = messages,                     # (3) conversation so far
    inferenceConfig = {"maxTokens": 16000},       # (4) inference knobs
)
```

### (1) System prompt

Sent verbatim on every call (see `agent.py` → `SYSTEM_PROMPT`):

```text
You are a helpful support engineer assistant.
You have access to local tools that can look up customers, fetch tenant metrics,
search application logs, and perform arithmetic.
When a question requires data, call the appropriate tool.
Chain tools together when needed (e.g. get the customer first to find their
tenant_id, then fetch metrics/logs).
When you have enough information, produce a concise final answer grounded in
the tool results.
If a question does not need any tool, just answer directly.
```

Why this shape:

- **Role framing** ("support engineer") biases the answer style.
- **Tool inventory in prose** tells the model what capabilities exist,
  independent of the JSON schemas (the schemas below are the machine
  contract; this paragraph is the human intent).
- **Chaining hint** nudges the model to do `get_customer → get_tenant_metrics
  → search_logs` instead of asking the user for a tenant id.
- **"No tool needed" escape** stops the model from calling a tool for
  trivia questions like *"what does SQL stand for?"*.

### (2) Tool definitions (`toolConfig`)

Each tool is described using the Bedrock Converse tool spec: a **name**, a
**natural-language description** (what the model reads to decide when to
use it), and a **JSON Schema** for its arguments (what the model must
produce). Full definitions live in `tools.py → BEDROCK_TOOL_SPECS`;
condensed here:

```jsonc
{
  "tools": [
    { "toolSpec": {
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression (e.g. '125 * 4'). Use ONLY for math.",
        "inputSchema": { "json": {
          "type": "object",
          "properties": { "expression": { "type": "string" } },
          "required": ["expression"]
    }}}},

    { "toolSpec": {
        "name": "get_customer",
        "description": "Look up a customer by id (e.g. 'CUST1001') or name (e.g. 'Acme'). Returns tenant_id.",
        "inputSchema": { "json": {
          "type": "object",
          "properties": { "customer_id": { "type": "string" } },
          "required": ["customer_id"]
    }}}},

    { "toolSpec": {
        "name": "get_tenant_metrics",
        "description": "Return DB/app metrics for a tenant: connections, CPU%, memory%, error rate%, latency, status.",
        "inputSchema": { "json": {
          "type": "object",
          "properties": { "tenant_id": { "type": "string" } },
          "required": ["tenant_id"]
    }}}},

    { "toolSpec": {
        "name": "search_logs",
        "description": "Search the local mock log store for lines matching a keyword for a tenant.",
        "inputSchema": { "json": {
          "type": "object",
          "properties": {
            "tenant_id": { "type": "string" },
            "keyword":   { "type": "string" }
          },
          "required": ["tenant_id", "keyword"]
    }}}}
  ]
}
```

The model never sees the Python bodies of these tools — only the name,
description, and schema. Good descriptions are what make tool selection
work.

### (3) `messages` — how it grows across iterations

The messages array is the **only** thing that changes between iterations.
Each ReAct turn appends to it and re-sends the whole history.

**Iteration 1 — just the user question:**

```jsonc
[
  { "role": "user", "content": [{ "text": "Why is Acme having DB connection problems?" }] }
]
```

**After iteration 1** — Bedrock returns `stopReason: "tool_use"` and an
assistant message containing a `toolUse` block. The runtime appends the
assistant message, then executes the tool locally, then appends the tool
result as a user turn:

```jsonc
[
  { "role": "user", "content": [{ "text": "Why is Acme having DB connection problems?" }] },

  { "role": "assistant", "content": [
      { "toolUse": {
          "toolUseId": "tu_01ABC...",
          "name": "get_customer",
          "input": { "customer_id": "Acme" }
      }}
  ]},

  { "role": "user", "content": [
      { "toolResult": {
          "toolUseId": "tu_01ABC...",
          "content": [{ "json": {
              "customer_id": "CUST1001",
              "name": "Acme Corporation",
              "tenant_id": "TENANT-ACME",
              "plan": "Enterprise"
          }}]
      }}
  ]}
]
```

**Iteration 2** re-sends the full array above; the model now knows the
tenant id and typically calls `get_tenant_metrics`. Iteration 3 typically
calls `search_logs`. Iteration 4 has enough context, so Bedrock returns
`stopReason: "end_turn"` with a plain text block — that's the final
answer.

Key rules of thumb:

- Every `toolUse` **must** be paired (by `toolUseId`) with a
  `toolResult` in the very next user turn.
- Tool results are sent as JSON so the model can reason about structured
  fields (`active_db_connections: 198`) rather than parsing prose.
- Nothing about which tool to call is baked into `messages`; the model
  decides based on the system prompt + tool descriptions + conversation
  so far.

### (4) Inference config

```jsonc
{ "maxTokens": 16000 }
```

`maxTokens` is a hard ceiling on the generated response (per call, not
per session). Temperature is left at the model default; for a
deterministic demo, set `"temperature": 0.0`.

### LangGraph equivalent

`agent_langgraph.py` sends the same four pieces, just packaged
differently:

| Raw Converse (agent.py)              | LangGraph (agent_langgraph.py)                   |
| ------------------------------------ | ------------------------------------------------ |
| `system=[{"text": SYSTEM_PROMPT}]`   | First `SystemMessage(content=SYSTEM_PROMPT)`     |
| `toolConfig={"tools": [...]}`        | `create_react_agent(llm, tools=LC_TOOLS)`        |
| `messages=[user, assistant, ...]`    | `HumanMessage / AIMessage / ToolMessage` list    |
| `inferenceConfig={"maxTokens": ...}` | `ChatBedrockConverse(max_tokens=..., temperature=...)` |

Under the hood LangGraph translates its typed messages back into
Bedrock's `toolUse` / `toolResult` blocks — same wire format.

## Primary demo scenario

> **Why is the Acme tenant experiencing database connection problems?**

Expected (LLM-driven) flow:

```text
get_customer(Acme) → get_tenant_metrics(TENANT-ACME) → search_logs(TENANT-ACME, "connection")
   → final diagnosis grounded in the tool results
```

The exact ordering (and even which tools to call) is chosen by the LLM
based on the tool descriptions — nothing about the sequence is hardcoded
in the agent.

## Additional demo scenarios

| Question                                          | Expected tools               |
| ------------------------------------------------- | ---------------------------- |
| `What is 125 * 4?`                                | `calculate`                  |
| `Tell me about customer CUST1001.`                | `get_customer`               |
| `Why is Acme having database connection problems?`| `get_customer` → `get_tenant_metrics` → `search_logs` |
| `In one sentence, what does SQL stand for?`       | *(no tool needed)*           |

## `agent.py` vs `agent_langgraph.py`

|                      | `agent.py`                                    | `agent_langgraph.py`                    |
| -------------------- | --------------------------------------------- | --------------------------------------- |
| Framework            | None — raw `boto3` + Bedrock Converse         | LangGraph `create_react_agent`          |
| Tool schemas         | Hand-written JSON schemas in `tools.py`       | Derived from `@tool`-decorated wrappers |
| Loop control         | Explicit `for iteration in range(...)` loop   | Handled by the LangGraph runtime        |
| Where to look first  | ✅ Best for learning                          | Good for seeing what a framework hides  |

Both call the **same** local Python functions and read the **same** mock
data.

## Notes / constraints

- Uses only Amazon Bedrock (LLM). No CloudWatch, Lambda, DynamoDB, S3,
  OpenSearch, or vector databases.
- No chain-of-thought is exposed — the console only shows tool selection,
  arguments, results, and the final answer.
- If SSO credentials expire mid-run, the client will attempt one
  `aws sso login` refresh automatically.
