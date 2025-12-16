import os
import json
import requests
from typing import Dict, Any, Iterable, Optional

ATLAS_URL = "https://api.atlascloud.ai/v1/chat/completions"

def _auth_headers() -> Dict[str, str]:
    key = os.getenv("ATLASCLOUD_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ATLASCLOUD_API_KEY manquante. Mets-la dans .env")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

def chat_stream(messages, model: Optional[str] = None, max_tokens: int = 2048, temperature: float = 0.2) -> Iterable[str]:
    """
    Stream type OpenAI: lignes 'data: {...}' + 'data: [DONE]'
    Renvoie les morceaux de texte (delta).
    """
    model = model or os.getenv("ATLAS_MODEL", "openai/gpt-oss-20b")
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    with requests.post(ATLAS_URL, headers=_auth_headers(), json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            # Plusieurs providers envoient parfois du JSON direct.
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
            else:
                # fallback si pas "data:"
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

            # Format type OpenAI: choices[0].delta.content
            try:
                delta = obj["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
            except Exception:
                # fallback: choices[0].message.content (non-stream)
                try:
                    content = obj["choices"][0]["message"]["content"]
                    if content:
                        yield content
                except Exception:
                    continue
