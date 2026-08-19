from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch

from .models import build_model, load_checkpoint


class DNNLabeler:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.model = build_model(config["class"], config.get("kwargs")).to(device)
        load_checkpoint(self.model, config["checkpoint"], device)
        self.model.eval()

    @torch.inference_mode()
    def label(self, features, raw_features=None, generated_labels=None) -> torch.Tensor:
        return self.model.run_inference(features)[1]


class DeviceListLabeler:
    """Uses the generated device-list matches shipped in UNSW-IoT CSVs."""

    def label(self, features, raw_features=None, generated_labels=None) -> torch.Tensor:
        if generated_labels is None:
            raise ValueError("device_list labeling requires generated_labels")
        return generated_labels


class OllamaLabeler:
    """Local LLM labeler backed by Ollama; no cloud API is used."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        del device
        self.host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "llama2:7b")
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", "300"))
        self.batch_size = int(os.getenv("OLLAMA_BATCH_SIZE", "10"))
        self.feature_names = config.get("feature_names", [])
        if self.batch_size < 1:
            raise ValueError("OLLAMA_BATCH_SIZE must be at least 1")

    def label(self, features, raw_features=None, generated_labels=None) -> torch.Tensor:
        values = raw_features if raw_features is not None else features
        rows = values.detach().cpu().tolist()
        labels: list[int] = []
        for start in range(0, len(rows), self.batch_size):
            labels.extend(self._label_batch(rows[start:start + self.batch_size]))
        return torch.tensor(labels, dtype=torch.long, device=values.device)

    def _label_batch(self, rows: list[list[float]]) -> list[int]:
        prompt = _classification_prompt(rows, self.feature_names)
        schema = _labels_schema(len(rows))
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a network intrusion detection classifier."},
                {"role": "user", "content": prompt},
            ],
            "format": schema,
            # Llama 2 often pretty-prints JSON, so allow enough tokens for
            # whitespace and line breaks in addition to the labels themselves.
            "options": {"temperature": 0, "seed": 42, "num_predict": max(128, len(rows) * 12)},
        }
        request = Request(
            f"{self.host}/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Cannot reach local Ollama at {self.host}. Start it with 'ollama serve'."
            ) from exc
        try:
            labels = json.loads(result["message"]["content"])["labels"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid structured response from {self.model}: {result}") from exc
        return _validate_labels(labels, len(rows), self.model)


class OpenAILabeler:
    """Cloud labeler using OpenAI's Responses API and strict structured output."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        del device
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required when labeler.type is 'openai'")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
        self.timeout = float(os.getenv("OPENAI_TIMEOUT", "300"))
        self.batch_size = int(os.getenv("OPENAI_BATCH_SIZE", "10"))
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
        self.feature_names = config.get("feature_names", [])
        if self.batch_size < 1 or self.max_retries < 0:
            raise ValueError("OPENAI_BATCH_SIZE must be positive and OPENAI_MAX_RETRIES nonnegative")

    def label(self, features, raw_features=None, generated_labels=None) -> torch.Tensor:
        values = raw_features if raw_features is not None else features
        rows = values.detach().cpu().tolist()
        labels: list[int] = []
        for start in range(0, len(rows), self.batch_size):
            labels.extend(self._label_batch(rows[start:start + self.batch_size]))
        return torch.tensor(labels, dtype=torch.long, device=values.device)

    def _label_batch(self, rows: list[list[float]]) -> list[int]:
        schema = _labels_schema(len(rows))
        payload = {
            "model": self.model,
            "store": False,
            "input": [
                {"role": "system", "content": "You are a network intrusion detection classifier."},
                {"role": "user", "content": _classification_prompt(rows, self.feature_names)},
            ],
            "text": {"format": {
                "type": "json_schema", "name": "flow_labels", "strict": True, "schema": schema,
            }},
            "max_output_tokens": max(128, len(rows) * 12),
        }
        request = Request(
            f"{self.base_url}/responses", data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        result = None
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                break
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self.max_retries:
                    raise RuntimeError(f"OpenAI Responses API request failed with HTTP {exc.code}") from exc
            except (URLError, TimeoutError) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError("Could not reach the OpenAI Responses API") from exc
            time.sleep(2 ** attempt)
        try:
            content = next(
                item["text"]
                for output in result["output"] if output.get("type") == "message"
                for item in output.get("content", []) if item.get("type") == "output_text"
            )
            labels = json.loads(content)["labels"]
        except (KeyError, StopIteration, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenAI model {self.model} returned no valid label output") from exc
        return _validate_labels(labels, len(rows), self.model)


def _classification_prompt(rows: list[list[float]], feature_names: list[str]) -> str:
    header = ",".join(feature_names) if feature_names else "features"
    flow_lines = "\n".join(
        f"{index}: " + ",".join(f"{value:.8g}" for value in row)
        for index, row in enumerate(rows, start=1)
    )
    return (
        "Classify every UNSW-NB15 network flow below. Label benign traffic 0 and "
        "malicious traffic 1. Keep input order. Return exactly one label per flow "
        f"in the required labels array.\nColumns: {header}\n{flow_lines}"
    )


def _labels_schema(count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"labels": {
            "type": "array", "items": {"type": "integer", "enum": [0, 1]},
            "minItems": count, "maxItems": count,
        }},
        "required": ["labels"],
        "additionalProperties": False,
    }


def _validate_labels(labels: Any, count: int, model: str) -> list[int]:
    if not isinstance(labels, list) or len(labels) != count or any(label not in (0, 1) for label in labels):
        raise RuntimeError(f"Expected {count} binary labels from {model}, received: {labels}")
    return [int(label) for label in labels]


def build_labeler(config: dict[str, Any], device: torch.device):
    kind = config["type"].lower()
    if kind == "dnn":
        return DNNLabeler(config, device)
    if kind == "device_list":
        return DeviceListLabeler()
    if kind in {"llm", "ollama"}:
        return OllamaLabeler(config, device)
    if kind == "openai":
        return OpenAILabeler(config, device)
    raise ValueError(f"Unsupported DRIFTADAPT labeler: {kind}")
