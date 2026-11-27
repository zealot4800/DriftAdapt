from __future__ import annotations

from collections import deque
import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import torch
from ..models import form_training_set, fit_model, train_classifier
from ..drift import RetrainingTrigger
from ..metrics import macro_f1

ABSTAIN = -1


def _label_batches(self, features, raw_features=None) -> torch.Tensor:
    values = raw_features if raw_features is not None else features
    rows = values.detach().cpu().tolist()
    labels: list[int] = []
    for start in range(0, len(rows), self.batch_size):
        labels.extend(self._label_batch(rows[start:start + self.batch_size]))
    return torch.tensor(
        labels, dtype=torch.long, device=features.device,
    )


class DNNLabeler:
    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def label(self, features, raw_features=None):
        return self.model.run_inference(features)[1]


class OllamaLabeler:
    """Local LLM labeler backed by Ollama; no cloud API is used."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.host = config.get("host", "http://127.0.0.1:11434").rstrip("/")
        self.model = config.get("model", "llama2:7b")
        self.timeout = float(config.get("timeout", 300))
        self.batch_size = int(config.get("batch_size", 10))
        self.feature_names = config.get("feature_names", [])
        self.seed = int(config.get("seed", 42))
        if self.batch_size < 1:
            raise ValueError("OLLAMA_BATCH_SIZE must be at least 1")

    label = _label_batches

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
            "options": {"temperature": 0, "seed": self.seed, "num_predict": max(128, len(rows) * 24)},
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
            output = json.loads(result["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid structured response from {self.model}") from exc
        return _validate_labels(output.get("labels") if isinstance(output, dict) else None, len(rows), self.model)


class OpenAILabeler:
    """Cloud labeler using OpenAI's Responses API and strict structured output."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required when labeler.type is 'openai'")
        self.base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.model = config.get("model", "gpt-5.4-nano")
        self.timeout = float(config.get("timeout", 300))
        self.batch_size = int(config.get("batch_size", 10))
        self.max_retries = int(config.get("max_retries", 3))
        self.feature_names = config.get("feature_names", [])
        if self.batch_size < 1 or self.max_retries < 0:
            raise ValueError("OPENAI_BATCH_SIZE must be positive and OPENAI_MAX_RETRIES nonnegative")

    label = _label_batches

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
                "type": "json_schema", "name": "window_labels", "strict": True, "schema": schema,
            }},
            "max_output_tokens": max(128, len(rows) * 24),
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
            output = json.loads(content)
        except (KeyError, StopIteration, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenAI model {self.model} returned no valid label output") from exc
        return _validate_labels(output.get("labels") if isinstance(output, dict) else None, len(rows), self.model)


def _classification_prompt(rows: list[list[float]], feature_names: list[str]) -> str:
    header = ",".join(feature_names) if feature_names else "features"
    window_lines = "\n".join(
        f"{index}: " + ",".join(f"{value:.8g}" for value in row)
        for index, row in enumerate(rows, start=1)
    )
    return (
        "Classify every packet window below. Label benign traffic 0 and "
        "malicious traffic 1. Keep input order. Return exactly one label per window "
        f"in the required array.\nColumns: {header}\n{window_lines}"
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
    if not isinstance(labels, list) or len(labels) != count or any(type(label) is not int or label not in (0, 1) for label in labels):
        raise RuntimeError(f"Expected {count} binary labels from {model}")
    return [int(label) for label in labels]


def build_labeler(config: dict[str, Any]):
    kind = config["type"].lower()
    if kind in {"llm", "ollama"}:
        return OllamaLabeler(config)
    if kind == "openai":
        return OpenAILabeler(config)
    raise ValueError(f"Unsupported CARAVAN labeler: {kind}")


class Caravan:
    unit = 'detection_window'

    def __init__(self, config):
        self.config = config
        self.buffer = deque(maxlen=config['stream']['buffer_windows'])
        self.version = 0
        self.optimizer = None
        self.teacher_samples = 0

    def train(self, data):
        self.model = train_classifier(self.config['model'], self.config, data)
        labeler = dict(self.config['labeler'], feature_names=data.feature_names,
                       seed=self.config['runtime']['seed'])
        self.labeler = (DNNLabeler(train_classifier(labeler, self.config, data))
                        if labeler['type'] == 'dnn' else build_labeler(labeler))
        settings = self.config['trigger']
        self.trigger = RetrainingTrigger(settings['threshold'], drop_threshold=settings['drop_threshold'],
                                        consecutive_windows=settings['consecutive_windows'])
        self.buffer.clear()
        self.version = self.teacher_samples = 0
        self.optimizer = None

    @torch.inference_mode()
    def infer(self, observation):
        return self.model.run_inference(observation['features'])[1]

    def adapt(self, observation, prediction):
        features = observation['features']
        produced = self.labeler.label(features, observation['raw_features'])
        self.teacher_samples += len(prediction)
        if (produced.dtype != torch.long or produced.shape != prediction.shape
                or not ((produced >= ABSTAIN) & (produced <= 1)).all()):
            raise ValueError('Labeler must return one binary label or -1 per packet window')
        valid = produced != ABSTAIN
        if not valid.any():
            return
        self.buffer.append((features[valid].detach().clone(), produced[valid].detach().clone(),
                            prediction[valid].detach().clone()))
        x, y, previous = (torch.cat([entry[index] for entry in self.buffer]) for index in range(3))
        proxy = macro_f1(y, previous)
        requested = self.trigger.update(proxy)
        x, y = form_training_set(x, y)
        if requested and x is not None:
            self.optimizer = fit_model(self.model, x, y, self.config['training'], optimizer=self.optimizer)
            self.version += 1

    def state(self):
        return {'adaptation_count': self.version, 'labeling_cost_samples': self.teacher_samples}
