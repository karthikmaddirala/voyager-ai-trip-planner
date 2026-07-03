"""
llm.py
──────
Handles all communication with Claude API.

Each phase (planner, replanner, synthesizer) uses a DIFFERENT system prompt,
but the same underlying LLM call mechanism.

The system prompt is sent on EVERY call because the LLM is stateless.
"""

import json
import time

from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, MODEL_NAME, MAX_TOKENS
from logutil import log

client = Anthropic(api_key=ANTHROPIC_API_KEY)


def run_agent_loop(system_prompt, user_text, tools, tool_impls, label="agent",
                   max_iters=8, temperature=None):
    """A REAL tool-use agent loop: the LLM is given the tools and DECIDES which to
    call. We execute its chosen calls, feed the results back, and repeat until the
    model produces a final (non-tool) answer. This is the agent/workflow line —
    here the MODEL drives the tool usage, not our code.
    `temperature` low (e.g. 0.2) keeps the candidate set stable run-to-run.
    Returns the final assistant text. `tool_impls` maps tool name -> python fn."""
    messages = [{"role": "user", "content": user_text}]
    kwargs = {"model": MODEL_NAME, "max_tokens": MAX_TOKENS, "system": system_prompt,
              "tools": tools}
    if temperature is not None:
        kwargs["temperature"] = temperature
    last = None
    for it in range(max_iters):
        last = client.messages.create(messages=messages, **kwargs)
        if last.stop_reason != "tool_use":
            log("AGENT✓", f"{label}: final answer (turn {it+1})")
            return extract_text(last)
        messages.append({"role": "assistant", "content": last.content})
        results = []
        for b in last.content:
            if b.type == "tool_use":
                log("AGENT⚙", f"{label}: {b.name}({json.dumps(b.input, default=str)[:90]})")
                try:
                    out = tool_impls[b.name](**b.input)
                except Exception as e:
                    out = {"error": str(e)}
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(out, default=str)[:7000]})
        messages.append({"role": "user", "content": results})
    log("AGENT", f"{label}: hit max_iters={max_iters}; returning last text")
    return extract_text(last)


def call_llm(messages, system_prompt, tools=None, temperature=None, label="llm"):
    """
    Call Claude with a specific system prompt for the current phase.

    Args:
        messages: conversation history
        system_prompt: phase-specific instructions (PLANNER / REPLANNER / SYNTHESIZER)
        tools: optional tool definitions (only used in some phases)
        temperature: optional sampling temperature (lower = more consistent output;
                     used by the stop proposer so marquee stops appear every run)

    BLOCKING — the caller waits here until response arrives.
    """
    log("LLM→", f"{label}: sys={len(system_prompt)}c msgs={len(messages)} "
                 f"tools={len(tools) if tools else 0} temp={temperature if temperature is not None else 'default'}")

    kwargs = {
        "model": MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "system": system_prompt,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    if temperature is not None:
        kwargs["temperature"] = temperature

    t0 = time.time()
    response = client.messages.create(**kwargs)
    out = sum(len(b.text) for b in response.content if b.type == "text")
    usage = getattr(response, "usage", None)
    toks = f" in={usage.input_tokens} out={usage.output_tokens}" if usage else ""
    log("LLM✓", f"{label}: {time.time()-t0:.1f}s stop={response.stop_reason} chars={out}{toks}")
    return response

def extract_text(response):
    """Extract concatenated text from a response's content blocks."""
    return "".join(b.text for b in response.content if b.type == "text")
