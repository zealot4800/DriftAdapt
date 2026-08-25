#!/usr/bin/env python3
"""Validated drift-conditioned repair support for the U55C host controller.

This module performs adapter inference only.  Adapter training remains in the
stimulation workflow.  A proposal contains sparse, absolute Q16.16 words for
the atomic selective-update protocol; it never contains a full model image.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stimulation.src.adapter import (  # noqa: E402
    AdapterBundle,
    load_adapter_bundle,
    predict_repair,
)
from stimulation.src.drift import jensen_shannon_divergence  # noqa: E402
from stimulation.src.drift_signature import (  # noqa: E402
    DRIFT_SIGNATURE_FORMAT_VERSION,
    SIGNATURE_FIELDS,
    _fixed_histogram_counts,
    build_drift_signature,
)
from stimulation.src.repair import apply_repair_plan  # noqa: E402


PARAMETER_KEYS = (
    "linear_relu_stack.0.weight",
    "linear_relu_stack.0.bias",
    "linear_relu_stack.2.weight",
    "linear_relu_stack.2.bias",
    "linear_relu_stack.4.weight",
    "linear_relu_stack.4.bias",
)
PARAMETER_COUNT = 182
PARAMETER_BITS = 32
PARAMETER_FRACTION_BITS = 16
FEATURE_BITS = 16
FEATURE_FRACTION_BITS = 8
PARAMETER_FORMAT = "signed Q16.16"

__all__ = (
    "DriftConditionedRuntime",
    "DriftConditionedUpdate",
    "FixedPointOutput",
    "PARAMETER_COUNT",
    "PARAMETER_KEYS",
    "apply_parameter_patches",
    "fixed_inference",
    "flatten_parameter_words",
)


@dataclass(frozen=True)
class FixedPointOutput:
    prediction: int
    logit0_q16: int
    logit1_q16: int
    margin_q16: int


@dataclass(frozen=True)
class DriftConditionedUpdate:
    proposal_id: int
    accepted: bool
    rejection_reason: str | None
    confidence_passed: bool | None
    adapter_confidence: float | None
    predicted_modules: tuple[str, ...]
    module_probabilities: dict[str, float]
    module_magnitudes: dict[str, float]
    signature_vector: tuple[float, ...]
    signature_per_feature: dict[str, dict[str, float]]
    signature_fields: tuple[str, ...]
    signature_format_version: int
    feature_drift_scores: dict[str, float | None]
    feature_drift_detected: bool
    drifted_feature_indices: tuple[int, ...]
    drifted_feature_names: tuple[str, ...]
    histogram_equivalent: bool | None
    patches: tuple[tuple[int, int], ...]
    patched_modules: tuple[str, ...]
    changed_parameter_count: int
    changed_parameter_percentage: float
    update_bytes: int
    software_changed_parameter_count: int
    software_changed_parameter_percentage: float
    drift_detection_latency_seconds: float
    signature_latency_seconds: float
    adapter_latency_seconds: float
    repair_latency_seconds: float
    latency_seconds: float
    parameter_format: str = PARAMETER_FORMAT


def _signed_32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def _saturate_s32(value: int) -> int:
    return max(-(1 << 31), min((1 << 31) - 1, value))


def _rounded_fixed(
    values: np.ndarray,
    *,
    fraction_bits: int,
    bits: int,
) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError("fixed-point input contains a non-finite value")
    scaled = np.rint(values.astype(np.float64) * (1 << fraction_bits)).astype(np.int64)
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if np.any(scaled < minimum) or np.any(scaled > maximum):
        bad = int(scaled[(scaled < minimum) | (scaled > maximum)][0])
        raise ValueError(f"fixed-point value {bad} does not fit signed {bits}-bit storage")
    return scaled


def flatten_parameter_words(model: nn.Module) -> tuple[int, ...]:
    """Flatten and quantize the exact 182-word U55C task-model layout."""
    state = model.state_dict()
    missing = [name for name in PARAMETER_KEYS if name not in state]
    if missing:
        raise ValueError(f"U55C task model is missing parameters: {', '.join(missing)}")
    values = np.concatenate([
        state[name].detach().cpu().numpy().reshape(-1)
        for name in PARAMETER_KEYS
    ])
    if values.size != PARAMETER_COUNT:
        raise ValueError(
            f"U55C task model must contain {PARAMETER_COUNT} flattened parameters; "
            f"received {values.size}"
        )
    fixed = _rounded_fixed(
        values, fraction_bits=PARAMETER_FRACTION_BITS, bits=PARAMETER_BITS,
    )
    return tuple(int(value) & 0xFFFFFFFF for value in fixed)


def apply_parameter_patches(
    words: Sequence[int],
    patches: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    """Apply absolute index/value patches to a 182-word software image."""
    if len(words) != PARAMETER_COUNT:
        raise ValueError(f"U55C parameter image must contain {PARAMETER_COUNT} words")
    updated = [int(value) & 0xFFFFFFFF for value in words]
    seen: set[int] = set()
    for index, value in patches:
        if not 0 <= int(index) < PARAMETER_COUNT:
            raise ValueError(f"U55C parameter patch index is out of range: {index}")
        if int(index) in seen:
            raise ValueError(f"U55C parameter patch index is duplicated: {index}")
        seen.add(int(index))
        updated[int(index)] = int(value) & 0xFFFFFFFF
    return tuple(updated)


def _fixed_output(feature_q8: np.ndarray, parameter_words: Sequence[int]) -> FixedPointOutput:
    parameters = np.asarray([_signed_32(int(word)) for word in parameter_words], dtype=np.int64)
    layer1_weights = parameters[0:128].reshape(8, 16)
    layer1_bias = parameters[128:136]
    layer2_weights = parameters[136:168].reshape(4, 8)
    layer2_bias = parameters[168:172]
    layer3_weights = parameters[172:180].reshape(2, 4)
    layer3_bias = parameters[180:182]

    activations = [int(value) << 8 for value in feature_q8]
    hidden1: list[int] = []
    for neuron in range(8):
        accumulator = int(layer1_bias[neuron])
        for feature in range(16):
            accumulator += (
                int(layer1_weights[neuron, feature]) * activations[feature]
            ) >> PARAMETER_FRACTION_BITS
        hidden1.append(max(0, _saturate_s32(accumulator)))

    hidden2: list[int] = []
    for neuron in range(4):
        accumulator = int(layer2_bias[neuron])
        for feature in range(8):
            accumulator += (
                int(layer2_weights[neuron, feature]) * hidden1[feature]
            ) >> PARAMETER_FRACTION_BITS
        hidden2.append(max(0, _saturate_s32(accumulator)))

    logits: list[int] = []
    for neuron in range(2):
        accumulator = int(layer3_bias[neuron])
        for feature in range(4):
            accumulator += (
                int(layer3_weights[neuron, feature]) * hidden2[feature]
            ) >> PARAMETER_FRACTION_BITS
        logits.append(accumulator)
    return FixedPointOutput(
        prediction=int(logits[1] > logits[0]),
        logit0_q16=logits[0],
        logit1_q16=logits[1],
        margin_q16=_saturate_s32(abs(logits[1] - logits[0])),
    )


def fixed_inference(
    features: torch.Tensor | np.ndarray | Sequence[Sequence[float]],
    parameter_words: Sequence[int],
) -> tuple[FixedPointOutput, ...]:
    """Mirror the U55C fixed-point DNN for software/hardware equivalence."""
    if len(parameter_words) != PARAMETER_COUNT:
        raise ValueError(f"U55C parameter image must contain {PARAMETER_COUNT} words")
    if isinstance(features, torch.Tensor):
        feature_values = features.detach().cpu().numpy()
    else:
        feature_values = np.asarray(features)
    if feature_values.ndim != 2 or feature_values.shape[1] != 16:
        raise ValueError("U55C fixed inference requires an N x 16 feature matrix")
    feature_q8 = _rounded_fixed(
        feature_values, fraction_bits=FEATURE_FRACTION_BITS, bits=FEATURE_BITS,
    )
    return tuple(_fixed_output(row, parameter_words) for row in feature_q8)


def _parameter_modules() -> tuple[str, ...]:
    modules: list[str] = []
    for name, count in zip(PARAMETER_KEYS, (128, 8, 32, 4, 8, 2)):
        modules.extend([name.rsplit(".", 1)[0]] * count)
    if len(modules) != PARAMETER_COUNT:
        raise AssertionError("invalid U55C parameter-module layout")
    return tuple(modules)


PARAMETER_MODULES = _parameter_modules()


class DriftConditionedRuntime:
    """Stateful, training-free U55C drift-conditioned repair helper."""

    def __init__(
        self,
        adapter_checkpoint: Path,
        base_model: nn.Module,
        device: torch.device,
        *,
        feature_names: Sequence[str],
        bins: int,
        reference_windows: int,
        drift_threshold: float,
        consecutive_windows: int,
        confidence_threshold: float,
        module_threshold: float,
        normalized_shift_clip: float = 8.0,
        initial_parameter_words: Sequence[int] | None = None,
    ) -> None:
        if (
            bins < 2
            or reference_windows < 1
            or consecutive_windows < 1
            or normalized_shift_clip <= 0
        ):
            raise ValueError("invalid U55C drift-signature configuration")
        if not 0.0 <= drift_threshold <= 1.0:
            raise ValueError("drift_threshold must be between zero and one")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one")
        if not 0.0 <= module_threshold <= 1.0:
            raise ValueError("module_threshold must be between zero and one")
        self.device = device
        self.feature_names = list(feature_names)
        self.bins = int(bins)
        self.reference_windows = int(reference_windows)
        self.drift_threshold = float(drift_threshold)
        self.consecutive_windows = int(consecutive_windows)
        self.confidence_threshold = float(confidence_threshold)
        self.module_threshold = float(module_threshold)
        self.normalized_shift_clip = float(normalized_shift_clip)
        self.signature_config = {
            "bins": self.bins,
            "reference_windows": self.reference_windows,
            "normalized_shift_clip": self.normalized_shift_clip,
        }
        self.base_model = copy.deepcopy(base_model).to(device).eval()
        self._active_model = copy.deepcopy(self.base_model).eval()
        self.bundle: AdapterBundle = load_adapter_bundle(
            Path(adapter_checkpoint).resolve(),
            self.base_model,
            device,
            feature_names=self.feature_names,
            signature_config=self.signature_config,
            confidence_threshold=self.confidence_threshold,
            module_threshold=self.module_threshold,
        )
        base_words = flatten_parameter_words(self.base_model)
        if initial_parameter_words is not None:
            initial = tuple(int(value) & 0xFFFFFFFF for value in initial_parameter_words)
            if initial != base_words:
                raise ValueError(
                    "active U55C parameter image does not match the validated base model"
                )
        self._active_words = base_words
        self._proposal_sequence = 0
        self._feature_drift_streaks = [0] * len(self.feature_names)
        self._drift_event_handled = False
        self._pending_update: DriftConditionedUpdate | None = None
        self._pending_candidate: nn.Module | None = None
        self._pending_words: tuple[int, ...] | None = None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        config_base: Path,
        base_model: nn.Module,
        device: torch.device,
        *,
        initial_parameter_words: Sequence[int] | None = None,
    ) -> "DriftConditionedRuntime":
        conditioned = dict(
            config.get("drift_conditioned")
            or config.get("drift_adapter")
            or {}
        )
        checkpoint_value = conditioned.get("adapter_checkpoint") or conditioned.get(
            "checkpoint"
        )
        if not checkpoint_value:
            raise ValueError("drift-conditioned adapter checkpoint is required")
        checkpoint = Path(checkpoint_value).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (Path(config_base) / checkpoint).resolve()
        drift = dict(config.get("drift", {}))
        return cls(
            checkpoint,
            base_model,
            device,
            feature_names=list(config["dataset"]["feature_names"]),
            bins=int(drift.get("bins", 16)),
            reference_windows=int(drift.get("reference_windows", 10)),
            drift_threshold=float(drift.get("threshold", 0.10)),
            consecutive_windows=int(drift.get("consecutive_windows", 1)),
            confidence_threshold=float(conditioned.get("confidence_threshold", 0.50)),
            module_threshold=float(conditioned.get("module_threshold", 0.50)),
            normalized_shift_clip=float(conditioned.get("signature_shift_clip", 8.0)),
            initial_parameter_words=initial_parameter_words,
        )

    @property
    def active_parameter_words(self) -> tuple[int, ...]:
        return self._active_words

    @property
    def active_model(self) -> nn.Module:
        return copy.deepcopy(self._active_model).eval()

    def _histogram_state(
        self,
        histogram: Mapping[str, Any],
        current_features: torch.Tensor,
    ) -> dict[str, Any]:
        if bool(histogram.get("overflow_error", False)):
            raise ValueError("U55C histogram monitor reported a counter overflow")
        if int(histogram.get("feature_count", -1)) != len(self.feature_names):
            raise ValueError("U55C histogram feature count does not match the adapter")
        if int(histogram.get("bin_count", -1)) != self.bins:
            raise ValueError("U55C histogram bin count does not match the adapter")
        if int(histogram.get("reference_windows", -1)) != self.reference_windows:
            raise ValueError("U55C histogram reference-window count does not match the adapter")
        if "feature_names" in histogram and list(histogram["feature_names"]) != self.feature_names:
            raise ValueError("U55C histogram feature ordering does not match the adapter")

        features = torch.as_tensor(current_features, dtype=torch.float32).detach().cpu()
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError("U55C window features do not match the adapter feature count")
        reference_counts = torch.as_tensor(histogram["reference_counts"], dtype=torch.int64)
        current_counts = torch.as_tensor(histogram["current_counts"], dtype=torch.int64)
        expected_shape = (len(self.feature_names), self.bins)
        if (
            tuple(reference_counts.shape) != expected_shape
            or tuple(current_counts.shape) != expected_shape
        ):
            raise ValueError("U55C histogram counter matrix has the wrong dimensions")
        if bool(torch.any(reference_counts < 0)) or bool(torch.any(current_counts < 0)):
            raise ValueError("U55C histogram counters must be nonnegative")
        ranges = histogram.get("ranges_q8")
        if not isinstance(ranges, Sequence) or len(ranges) != len(self.feature_names):
            raise ValueError("U55C histogram range state has the wrong dimensions")
        try:
            minimum = torch.tensor(
                [_signed_32(int(pair[0])) / 256.0 for pair in ranges],
                dtype=torch.float32,
            )
            maximum = torch.tensor(
                [_signed_32(int(pair[1])) / 256.0 for pair in ranges],
                dtype=torch.float32,
            )
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError("U55C histogram range state is malformed") from error
        if bool(torch.any(maximum < minimum)):
            raise ValueError("U55C histogram range maximum precedes its minimum")

        window_id = int(histogram.get("window_id", -1))
        if window_id < 0:
            raise ValueError("U55C histogram window ID is missing or invalid")
        reference_ready = bool(
            histogram.get("reference_ready", False)
            and window_id >= self.reference_windows
        )
        histogram_equivalent: bool | None = None
        if reference_ready:
            reconstructed_counts = _fixed_histogram_counts(
                features, minimum, maximum, self.bins,
            )
            if not torch.equal(reconstructed_counts, current_counts):
                mismatched = torch.nonzero(
                    torch.any(reconstructed_counts != current_counts, dim=1),
                    as_tuple=False,
                ).flatten().tolist()
                raise ValueError(
                    "U55C histogram/current-feature mismatch for feature indices: "
                    + ", ".join(str(index) for index in mismatched)
                )
            histogram_equivalent = True
        scores: dict[str, float | None]
        if reference_ready:
            scores = {
                name: jensen_shannon_divergence(
                    reference_counts[index], current_counts[index],
                )
                for index, name in enumerate(self.feature_names)
            }
        else:
            scores = dict.fromkeys(self.feature_names)
        return {
            "reference_ready": reference_ready,
            "histogram_equivalent": histogram_equivalent,
            "features": features,
            "reference_counts": reference_counts,
            "current_counts": current_counts,
            "minimum": minimum,
            "maximum": maximum,
            "scores": scores,
        }

    def propose(
        self,
        histogram: Mapping[str, Any],
        current_features: torch.Tensor,
    ) -> DriftConditionedUpdate:
        """Create one base-relative sparse hardware proposal from feature state."""
        if self._pending_update is not None:
            raise RuntimeError(
                "accept or discard the pending U55C repair before proposing another"
            )
        started = time.perf_counter()
        detection_started = time.perf_counter()
        histogram_state = self._histogram_state(histogram, current_features)
        scores = histogram_state["scores"]
        if histogram_state["reference_ready"]:
            for index, name in enumerate(self.feature_names):
                score = scores[name]
                if score is not None and score > self.drift_threshold:
                    self._feature_drift_streaks[index] = min(
                        self._feature_drift_streaks[index] + 1,
                        self.consecutive_windows,
                    )
                else:
                    self._feature_drift_streaks[index] = 0
        else:
            self._feature_drift_streaks = [0] * len(self.feature_names)
        drifted_indices = tuple(
            index for index, streak in enumerate(self._feature_drift_streaks)
            if streak >= self.consecutive_windows
        )
        feature_drift_detected = bool(drifted_indices)
        if not feature_drift_detected:
            self._drift_event_handled = False
        detection_latency = time.perf_counter() - detection_started

        signature: dict[str, Any] = {
            "drift_signature_vector": [],
            "drift_signature_per_feature": {},
        }
        signature_latency = 0.0
        adapter_latency = 0.0
        repair_latency = 0.0
        confidence: float | None = None
        confidence_passed: bool | None = None
        predicted_modules: tuple[str, ...] = ()
        module_probabilities: dict[str, float] = {}
        module_magnitudes: dict[str, float] = {}
        patches: tuple[tuple[int, int], ...] = ()
        patched_modules: tuple[str, ...] = ()
        software_changed = 0
        software_changed_percentage = 0.0
        candidate: nn.Module | None = None
        candidate_words: tuple[int, ...] | None = None
        rejection_reason: str | None = None
        if not histogram_state["reference_ready"]:
            rejection_reason = "reference_not_ready"
        elif not feature_drift_detected:
            rejection_reason = "no_persistent_feature_drift"
        elif self._drift_event_handled:
            rejection_reason = "drift_event_already_handled"
        else:
            signature_started = time.perf_counter()
            signature = build_drift_signature(
                self.feature_names,
                histogram_state["reference_counts"],
                histogram_state["features"],
                histogram_state["minimum"],
                histogram_state["maximum"],
                scores,
                normalized_shift_clip=self.normalized_shift_clip,
            )
            signature_latency = time.perf_counter() - signature_started
            adapter_started = time.perf_counter()
            plan, prediction = predict_repair(
                self.bundle,
                signature["drift_signature_vector"],
                module_threshold=self.module_threshold,
                parameter_bits=PARAMETER_BITS,
            )
            adapter_latency = time.perf_counter() - adapter_started
            confidence = float(prediction["adapter_confidence"])
            confidence_passed = confidence >= self.confidence_threshold
            predicted_modules = tuple(prediction["predicted_modules"])
            module_probabilities = dict(prediction["adapter_module_probabilities"])
            module_magnitudes = dict(prediction["predicted_module_magnitudes"])
            if not confidence_passed:
                rejection_reason = "low_confidence"
            elif not plan.affected_modules:
                rejection_reason = "no_predicted_modules"
            else:
                repair_started = time.perf_counter()
                candidate, repair_metrics = apply_repair_plan(
                    self._active_model,
                    plan,
                    parameter_bits=PARAMETER_BITS,
                    anchor_model=self.base_model,
                )
                if not all(
                    bool(torch.isfinite(parameter).all())
                    for parameter in candidate.parameters()
                ):
                    raise ValueError("drift-conditioned repair produced non-finite parameters")
                candidate_words = flatten_parameter_words(candidate)
                patches = tuple(
                    (index, word)
                    for index, word in enumerate(candidate_words)
                    if word != self._active_words[index]
                )
                software_changed = int(repair_metrics["repair_changed_parameters"])
                software_changed_percentage = float(
                    repair_metrics["repair_changed_parameter_percentage"]
                )
                patched_modules = tuple(dict.fromkeys(
                    PARAMETER_MODULES[index] for index, _word in patches
                ))
                repair_latency = time.perf_counter() - repair_started
                if not patches:
                    rejection_reason = "no_quantized_parameter_change"

        accepted = rejection_reason is None
        self._proposal_sequence += 1
        changed = len(patches) if accepted else 0
        if not accepted:
            patches = ()
            patched_modules = ()
            candidate = None
            candidate_words = None
        update = DriftConditionedUpdate(
            proposal_id=self._proposal_sequence,
            accepted=accepted,
            rejection_reason=rejection_reason,
            confidence_passed=confidence_passed,
            adapter_confidence=confidence,
            predicted_modules=predicted_modules,
            module_probabilities=module_probabilities,
            module_magnitudes=module_magnitudes,
            signature_vector=tuple(signature["drift_signature_vector"]),
            signature_per_feature=dict(signature["drift_signature_per_feature"]),
            signature_fields=tuple(SIGNATURE_FIELDS),
            signature_format_version=DRIFT_SIGNATURE_FORMAT_VERSION,
            feature_drift_scores=scores,
            feature_drift_detected=feature_drift_detected,
            drifted_feature_indices=drifted_indices,
            drifted_feature_names=tuple(
                self.feature_names[index] for index in drifted_indices
            ),
            histogram_equivalent=histogram_state["histogram_equivalent"],
            patches=patches,
            patched_modules=patched_modules,
            changed_parameter_count=changed,
            changed_parameter_percentage=100.0 * changed / PARAMETER_COUNT,
            update_bytes=changed * (PARAMETER_BITS // 8),
            software_changed_parameter_count=software_changed,
            software_changed_parameter_percentage=software_changed_percentage,
            drift_detection_latency_seconds=detection_latency,
            signature_latency_seconds=signature_latency,
            adapter_latency_seconds=adapter_latency,
            repair_latency_seconds=repair_latency,
            latency_seconds=time.perf_counter() - started,
        )
        if accepted:
            self._pending_update = update
            self._pending_candidate = candidate
            self._pending_words = candidate_words
        return update

    def accept(self, update: DriftConditionedUpdate) -> None:
        """Advance software state only after the FPGA atomically commits patches."""
        candidate_words = self.candidate_parameter_words(update)
        if self._pending_candidate is None:
            raise RuntimeError("U55C repair proposal lost its candidate model")
        self._active_model = self._pending_candidate.eval()
        self._active_words = candidate_words
        self._drift_event_handled = True
        self.discard()

    def candidate_parameter_words(
        self,
        update: DriftConditionedUpdate,
    ) -> tuple[int, ...]:
        """Verify and return pending words without advancing committed state."""
        if self._pending_update is None or update.proposal_id != self._pending_update.proposal_id:
            raise ValueError("U55C repair update is not the active pending proposal")
        if update != self._pending_update or not update.accepted:
            raise ValueError("U55C repair update cannot be accepted")
        if self._pending_words is None:
            raise RuntimeError("U55C repair proposal lost its candidate state")
        reconstructed = apply_parameter_patches(self._active_words, update.patches)
        if reconstructed != self._pending_words:
            raise RuntimeError("U55C repair patches do not reproduce the software candidate")
        return reconstructed

    def discard(self) -> None:
        """Discard an uncommitted proposal without changing active/base state."""
        self._pending_update = None
        self._pending_candidate = None
        self._pending_words = None
