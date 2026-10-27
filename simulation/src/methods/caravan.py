from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import torch
from ..models import build_model, load_checkpoint, form_training_set, retrain_model
from ..drift import RetrainingTrigger
from ..metrics import macro_f1

ABSTAIN = -1


def _label_batches(self, features, raw_features=None, generated_labels=None) -> torch.Tensor:
    del generated_labels
    values = raw_features if raw_features is not None else features
    rows = values.detach().cpu().tolist()
    labels: list[int] = []
    for start in range(0, len(rows), self.batch_size):
        labels.extend(self._label_batch(rows[start:start + self.batch_size]))
    return torch.tensor(
        labels, dtype=torch.long, device=features.device,
    )


class DNNLabeler:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.model = build_model(config["class"], config.get("kwargs")).to(device)
        load_checkpoint(self.model, config["checkpoint"], device)
        self.model.eval()

    @torch.inference_mode()
    def label(self, features, raw_features=None, generated_labels=None) -> torch.Tensor:
        del raw_features, generated_labels
        return self.model.run_inference(features)[1]


class DeviceListLabeler:
    """Uses the generated device-list matches shipped in UNSW-IoT CSVs."""

    def label(self, features, raw_features=None, generated_labels=None) -> torch.Tensor:
        del features, raw_features
        if generated_labels is None:
            raise ValueError("device_list labeling requires generated_labels")
        return generated_labels


class OllamaLabeler:
    """Local LLM labeler backed by Ollama; no cloud API is used."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        del device
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
            raise RuntimeError(f"Invalid structured response from {self.model}: {result}") from exc
        return _validate_labels(output.get("labels"), len(rows), self.model)


class OpenAILabeler:
    """Cloud labeler using OpenAI's Responses API and strict structured output."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        del device
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required when labeler.type is 'openai'")
        self.base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.model = config.get("model", "gpt-5.4-nano")
        self.timeout = float(config.get("timeout", 300))
        self.batch_size = int(config.get("batch_size", 10))
        self.max_retries = int(config.get("max_retries", 3))
        self.feature_names = config.get("feature_names", [])
        self.seed = int(config.get("seed", 42))
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
                "type": "json_schema", "name": "flow_labels", "strict": True, "schema": schema,
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
        return _validate_labels(output.get("labels"), len(rows), self.model)


def _classification_prompt(rows: list[list[float]], feature_names: list[str]) -> str:
    header = ",".join(feature_names) if feature_names else "features"
    flow_lines = "\n".join(
        f"{index}: " + ",".join(f"{value:.8g}" for value in row)
        for index, row in enumerate(rows, start=1)
    )
    return (
        "Classify every network flow below. Label benign traffic 0 and "
        "malicious traffic 1. Keep input order. Return exactly one label per flow "
        f"in the required array.\nColumns: {header}\n{flow_lines}"
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



class Caravan:
    unit = 'flow'

    def __init__(self, config, output):
        self.config = config
        self.version = 0
        self.optimizer = None
        self.teacher_samples = 0
        self.labeling_seconds = 0.0
        self.update_bytes = 0

    def train(self, data):
        from ..models import prepare_student
        prepare_student(self.config['model'], self.config, data, train=True)
        if self.config['labeler']['type'] == 'dnn':
            prepare_student(self.config['labeler'], self.config, data, train=True)
        return {'source_training_rows': len(data.labels[0])}

    def prepare(self, data):
        from ..models import prepare_student
        self.model = prepare_student(self.config['model'], self.config, data)
        labeler = dict(self.config['labeler'], feature_names=data.feature_names, seed=self.config['runtime']['seed'])
        if labeler['type'] == 'dnn':
            self.labeler = DNNLabeler.__new__(DNNLabeler)
            self.labeler.model = prepare_student(labeler, self.config, data)
        else:
            if labeler['type'] in ('ollama', 'openai') and self.config['dataset']['num_classes'] != 2:
                raise ValueError('The existing LLM teacher supports binary labels only')
            self.labeler = build_labeler(labeler, data.device)
        settings = self.config['trigger']
        self.trigger = RetrainingTrigger(settings['threshold'], drop_threshold=settings['drop_threshold'],
                                        consecutive_windows=settings['consecutive_windows'])

    @torch.inference_mode()
    def infer(self, observation):
        return self.model.run_inference(observation['features'])[1]

    def adapt(self, observation, prediction):
        started = time.perf_counter()
        count = len(prediction)
        budget = observation.get('label_budget')
        if budget is not None:
            count = min(count, budget)
        if count == 0:
            return {'adapted':False, 'teacher_samples':0, 'accepted_labels':0, 'model_update_bytes':0}
        features = observation['features'][:count]
        produced = self.labeler.label(features, observation['raw_features'][:count], observation['generated_labels'][:count])
        elapsed = time.perf_counter()-started
        self.labeling_seconds += elapsed
        self.teacher_samples += count
        valid = produced != ABSTAIN
        event = {'adapted': False, 'teacher_samples': count, 'accepted_labels': int(valid.sum()),
                 'labeling_seconds': elapsed, 'model_update_bytes': 0}
        if not valid.any():
            return event
        proxy = macro_f1(produced[valid], prediction[:count][valid])
        requested = self.trigger.update(proxy)
        settings = self.config['training']
        x, y = form_training_set(features[valid], produced[valid],
                                 balance_binary=settings['balance_binary'])
        event['proxy_f1'] = proxy
        if requested and x is not None:
            elapsed, self.optimizer = retrain_model(self.model, x, y, epochs=settings['epochs'],
                batch_size=settings['batch_size'], learning_rate=settings['learning_rate'],
                optimizer_name=settings['optimizer'], optimizer=self.optimizer)
            self.version += 1
            size = sum(p.numel() for p in self.model.parameters()) * self.config['runtime']['parameter_bits']
            size = (size+7)//8
            self.update_bytes += size
            event.update(adapted=True, training_seconds=elapsed, model_update_bytes=size)
        return event

    def state(self):
        return {'adaptation_count': self.version, 'labeling_cost_samples': self.teacher_samples,
                'labeling_seconds': self.labeling_seconds, 'model_update_bytes': self.update_bytes}
