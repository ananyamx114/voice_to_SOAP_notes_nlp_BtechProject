import json
import os
import re
from openai import OpenAI
from .config import env_value


def parse_json(text: str) -> dict:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response did not contain a JSON object")
        result = json.loads(cleaned[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("LLM response JSON must be an object")
    return result


class LocalLLM:
    def __init__(self, config: dict):
        self.model = env_value(config, "model_env", config.get("default_model"))
        base_url = env_value(config, "base_url_env", config.get("default_base_url"))
        api_key = env_value(config, "api_key_env", "ollama")
        if not self.model or not base_url or not api_key:
            raise RuntimeError("Local LLM configuration is incomplete")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.temperature = float(config.get("temperature", 0))
        self.max_tokens = int(config.get("max_tokens", 1800))

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(f"LLM {self.model!r} returned an empty response")
        return content

    def complete_json(self, system: str, user: str) -> dict:
        try:
            return parse_json(self.complete(system, user, json_mode=True))
        except Exception:
            # Some local OpenAI-compatible servers do not implement response_format.
            raw = self.complete(system, user, json_mode=False)
            return parse_json(raw)
