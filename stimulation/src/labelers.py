from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch

from .models import build_model, load_checkpoint


ABSTAIN = -1


class DNNLabeler:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.model = build_model(config["class"], config.get("kwargs")).to(device)
        load_checkpoint(self.model, config["checkpoint"], device)
        self.model.eval()

    @torch.inference_mode()
    def label(self, features, raw_features=None, generated_labels=None) -> torch.Tensor:
        return self.model.run_inference(features)[1]

    @torch.inference_mode()
    def label_with_metadata(self, features, raw_features=None, generated_labels=None):
        del raw_features, generated_labels
        _, representation_module = list(self.model.named_hidden_modules().items())[-1]
        captured = {}

        def hook(_module, _inputs, output):
            captured["embedding"] = output.clamp_min(0)

        handle = representation_module.register_forward_hook(hook)
        try:
            scores = self.model(features)
        finally:
            handle.remove()
        confidence, labels = scores.max(dim=1)
        return labels, confidence, captured["embedding"]


class DeviceListLabeler:
    """Uses the generated device-list matches shipped in UNSW-IoT CSVs."""

    def label(self, features, raw_features=None, generated_labels=None) -> torch.Tensor:
        if generated_labels is None:
            raise ValueError("device_list labeling requires generated_labels")
        return generated_labels

    def label_with_metadata(self, features, raw_features=None, generated_labels=None):
        labels = self.label(features, raw_features, generated_labels)
        confidence = (labels != ABSTAIN).to(features.dtype)
        return labels, confidence, features


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
        labels, _, _ = self.label_with_metadata(features, raw_features, generated_labels)
        return labels

    def label_with_metadata(self, features, raw_features=None, generated_labels=None):
        del generated_labels
        values = raw_features if raw_features is not None else features
        rows = values.detach().cpu().tolist()
        labels: list[int] = []
        confidences: list[float] = []
        for start in range(0, len(rows), self.batch_size):
            batch_labels, batch_confidences = self._label_batch(rows[start:start + self.batch_size])
            labels.extend(batch_labels)
            confidences.extend(batch_confidences)
        return (
            torch.tensor(labels, dtype=torch.long, device=features.device),
            torch.tensor(confidences, dtype=features.dtype, device=features.device),
            features,
        )

    def _label_batch(self, rows: list[list[float]]) -> tuple[list[int], list[float]]:
        prompt = _classification_prompt(rows, self.feature_names)
        schema = _teacher_schema(len(rows))
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
            "options": {"temperature": 0, "seed": 42, "num_predict": max(128, len(rows) * 24)},
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
        return _validate_teacher_output(output, len(rows), self.model)


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
        labels, _, _ = self.label_with_metadata(features, raw_features, generated_labels)
        return labels

    def label_with_metadata(self, features, raw_features=None, generated_labels=None):
        del generated_labels
        values = raw_features if raw_features is not None else features
        rows = values.detach().cpu().tolist()
        labels: list[int] = []
        confidences: list[float] = []
        for start in range(0, len(rows), self.batch_size):
            batch_labels, batch_confidences = self._label_batch(rows[start:start + self.batch_size])
            labels.extend(batch_labels)
            confidences.extend(batch_confidences)
        return (
            torch.tensor(labels, dtype=torch.long, device=features.device),
            torch.tensor(confidences, dtype=features.dtype, device=features.device),
            features,
        )

    def _label_batch(self, rows: list[list[float]]) -> tuple[list[int], list[float]]:
        schema = _teacher_schema(len(rows))
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
        return _validate_teacher_output(output, len(rows), self.model)


def _classification_prompt(rows: list[list[float]], feature_names: list[str]) -> str:
    header = ",".join(feature_names) if feature_names else "features"
    flow_lines = "\n".join(
        f"{index}: " + ",".join(f"{value:.8g}" for value in row)
        for index, row in enumerate(rows, start=1)
    )
    return (
        "Classify every network flow below. Label benign traffic 0 and "
        "malicious traffic 1. Keep input order. Return exactly one label per flow "
        "and a calibrated confidence from 0 to 1 for each label in the required "
        f"arrays.\nColumns: {header}\n{flow_lines}"
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


def _teacher_schema(count: int) -> dict[str, Any]:
    schema = _labels_schema(count)
    schema["properties"]["confidences"] = {
        "type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
        "minItems": count, "maxItems": count,
    }
    schema["required"].append("confidences")
    return schema


def _validate_labels(labels: Any, count: int, model: str) -> list[int]:
    if not isinstance(labels, list) or len(labels) != count or any(label not in (0, 1) for label in labels):
        raise RuntimeError(f"Expected {count} binary labels from {model}, received: {labels}")
    return [int(label) for label in labels]


def _validate_teacher_output(
    output: Any, count: int, model: str,
) -> tuple[list[int], list[float]]:
    if not isinstance(output, dict):
        raise RuntimeError(f"Expected structured labels and confidences from {model}")
    labels = _validate_labels(output.get("labels"), count, model)
    confidences = output.get("confidences")
    if (
        not isinstance(confidences, list)
        or len(confidences) != count
        or any(not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in confidences)
    ):
        raise RuntimeError(f"Expected {count} confidence values from {model}")
    return labels, [float(value) for value in confidences]


class HierarchicalLabeler:
    """Teacher knownness gate with a trusted fallback and binary abstention."""

    def __init__(self, teacher, config: dict[str, Any], num_classes: int) -> None:
        self.teacher = teacher
        self.confidence_threshold = float(config.get("confidence_threshold", 0.95))
        self.ood_threshold = float(config.get("ood_threshold", 0.30))
        self.fallback = str(config.get("fallback", "oracle")).lower()
        self.num_classes = num_classes
        self.reference_mean: torch.Tensor | None = None
        self.reference_variance: torch.Tensor | None = None
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("labeling.confidence_threshold must be between 0 and 1")
        if not 0.0 <= self.ood_threshold <= 1.0:
            raise ValueError("labeling.ood_threshold must be between 0 and 1")
        if self.fallback not in {"oracle", "abstain"}:
            raise ValueError("labeling.fallback must be oracle or abstain")

    @torch.inference_mode()
    def label(
        self,
        features: torch.Tensor,
        raw_features: torch.Tensor,
        generated_labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict]:
        started = time.perf_counter()
        count = features.shape[0]
        if count == 0:
            empty = torch.empty(0, dtype=torch.long, device=features.device)
            return empty, self._metrics(0, 0, 0, 0, 0, started, [], [], [], [])

        teacher_labels, confidence, embeddings = self.teacher.label_with_metadata(
            features, raw_features, generated_labels,
        )
        embeddings = embeddings.detach().to(torch.float32)
        if self.reference_mean is None:
            self.reference_mean = embeddings.mean(dim=0)
            self.reference_variance = embeddings.var(dim=0, unbiased=False).clamp_min(1e-6)
        squared_distance = (
            (embeddings - self.reference_mean).square() / self.reference_variance
        ).mean(dim=1)
        ood_scores = squared_distance / (1.0 + squared_distance)
        valid_class = (teacher_labels >= 0) & (teacher_labels < self.num_classes)
        known = (
            (confidence >= self.confidence_threshold)
            & (ood_scores <= self.ood_threshold)
            & valid_class
        )
        trusted = torch.full_like(teacher_labels, ABSTAIN)
        trusted[known] = teacher_labels[known]
        unknown = ~known
        metrics = self._metrics(
            count,
            int(known.sum().item()),
            int(unknown.sum().item()),
            0,
            int(unknown.sum().item()),
            started,
            teacher_labels.detach().cpu().tolist(),
            confidence.detach().cpu().tolist(),
            ood_scores.detach().cpu().tolist(),
            known.detach().cpu().tolist(),
        )
        return trusted, metrics

    def apply_fallback(
        self,
        teacher_labels: torch.Tensor,
        oracle_labels: torch.Tensor,
        metrics: dict,
    ) -> tuple[torch.Tensor, dict]:
        """Resolve pre-declared teacher unknowns without exposing oracle labels upstream."""
        started = time.perf_counter()
        trusted = teacher_labels.clone()
        unknown = trusted == ABSTAIN
        fallback_count = 0
        if self.fallback == "oracle":
            trusted[unknown] = oracle_labels[unknown]
            fallback_count = int(unknown.sum().item())
        updated = dict(metrics)
        updated["fallback_labeled_samples"] = fallback_count
        updated["abstained_samples"] = int((trusted == ABSTAIN).sum().item())
        updated["accepted_label_count"] = int((trusted != ABSTAIN).sum().item())
        updated["labeling_time"] += time.perf_counter() - started
        return trusted, updated

    def _metrics(
        self,
        sent: int,
        known: int,
        unknown: int,
        fallback: int,
        abstained: int,
        started: float,
        teacher_labels: list,
        confidence: list,
        ood_scores: list,
        known_decisions: list,
    ) -> dict:
        return {
            "samples_sent_to_teacher": sent,
            "teacher_known_samples": known,
            "teacher_unknown_samples": unknown,
            "fallback_labeled_samples": fallback,
            "abstained_samples": abstained,
            "teacher_coverage": known / sent if sent else 0.0,
            "accepted_label_count": sent - abstained,
            "labeling_time": time.perf_counter() - started,
            "teacher_predicted_labels": teacher_labels,
            "teacher_confidences": confidence,
            "teacher_ood_scores": ood_scores,
            "teacher_known_decisions": known_decisions,
        }


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
