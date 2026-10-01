"""
Small helper for talking to Ollama (the free AI model running on your laptop).

Ollama runs a local web server at http://localhost:11434. We send it a message
and ask it to reply in JSON so our code can read the answer reliably.

Because the model runs on YOUR computer, the notes never leave your machine.
That is the same reason hospitals prefer local or approved models for real
patient data (HIPAA).
"""
import json
import re

import requests

from config import OLLAMA_MODEL, OLLAMA_URL


class OllamaError(Exception):
    pass


def check_ollama(model=OLLAMA_MODEL):
    """Stop early with a friendly message if Ollama isn't ready."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
    except requests.RequestException:
        raise OllamaError(
            "Could not reach Ollama at " + OLLAMA_URL + ".\n"
            "  -> Open the Ollama app (or run 'ollama serve' in another terminal) and try again."
        )
    installed = [m["name"] for m in r.json().get("models", [])]
    wanted = model if ":" in model else model + ":latest"
    if wanted not in installed:
        raise OllamaError(
            f"The model '{model}' is not downloaded yet.\n"
            f"  -> Run this in your terminal:  ollama pull {model}\n"
            f"  Models you have now: {installed or 'none'}"
        )


def chat_json(system_prompt, user_prompt, model=OLLAMA_MODEL, timeout=300):
    """Send one message to the model and return its reply as a Python dict."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",          # force a JSON reply
        "stream": False,
        "options": {"temperature": 0},  # 0 = same answer every time (reproducible)
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise OllamaError(f"Request to Ollama failed: {e}")
    if r.status_code != 200:
        raise OllamaError(f"Ollama returned an error ({r.status_code}): {r.text[:300]}")

    content = r.json()["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Sometimes small models add extra text; grab the first {...} block.
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"_raw": content}
