from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def _two_hidden_stack(input_size: int, first: int, second: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, first), nn.ReLU(),
        nn.Linear(first, second), nn.ReLU(),
        nn.Linear(second, output_size), nn.Softmax(dim=1),
    )


class InferenceModel(nn.Module):
    def run_inference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.forward(x)
        return scores, torch.argmax(scores, dim=1)

    def named_hidden_modules(self) -> dict[str, nn.Linear]:
        """Return hidden layers by their checkpoint-compatible module names."""
        linear_modules = [
            (name, module) for name, module in self.named_modules()
            if isinstance(module, nn.Linear)
        ]
        return dict(linear_modules[:-1])

    def named_classifier_module(self) -> tuple[str, nn.Linear]:
        """Return the final classifier and its checkpoint-compatible name."""
        linear_modules = [
            (name, module) for name, module in self.named_modules()
            if isinstance(module, nn.Linear)
        ]
        if not linear_modules:
            raise ValueError("An adaptation model must contain at least one linear layer")
        return linear_modules[-1]


class ID_CIC_IDS2017_small_pforest(InferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear_relu_stack = _two_hidden_stack(16, 8, 4, 2)
        self.linear_relu_stack.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_relu_stack(x)


class ElasticTwoHiddenLayerModel(InferenceModel):
    """Software-only two-hidden-layer model with mask-controlled capacity.

    Dataset subclasses define the input, output, and minimum/maximum hidden
    widths. Linear modules retain a shared Sequential layout so impact mapping,
    repair generation, and capacity training use one implementation.
    """

    min_layer1_neurons = 8
    min_layer2_neurons = 4
    max_layer1_neurons = 16
    max_layer2_neurons = 8
    input_features = 16
    output_classes = 2
    legacy_checkpoint_prefix = "linear_relu_stack"

    _LEGACY_SHAPES = {
        "linear_relu_stack.0.weight": (8, 16),
        "linear_relu_stack.0.bias": (8,),
        "linear_relu_stack.2.weight": (4, 8),
        "linear_relu_stack.2.bias": (4,),
        "linear_relu_stack.4.weight": (2, 4),
        "linear_relu_stack.4.bias": (2,),
    }

    def __init__(self) -> None:
        super().__init__()
        self.linear_relu_stack = _two_hidden_stack(
            self.input_features, self.max_layer1_neurons,
            self.max_layer2_neurons, self.output_classes,
        )
        self.linear_relu_stack.apply(_init_weights)
        self.register_buffer(
            "layer1_mask",
            torch.tensor(
                [True] * self.min_layer1_neurons
                + [False] * (self.max_layer1_neurons - self.min_layer1_neurons)
            ),
        )
        self.register_buffer(
            "layer2_mask",
            torch.tensor(
                [True] * self.min_layer2_neurons
                + [False] * (self.max_layer2_neurons - self.min_layer2_neurons)
            ),
        )
        self.register_buffer(
            "layer1_clone_sources",
            torch.full((self.max_layer1_neurons,), -1, dtype=torch.long),
        )
        self.register_buffer(
            "layer2_clone_sources",
            torch.full((self.max_layer2_neurons,), -1, dtype=torch.long),
        )
        # Capacity retraining may update legacy-active neurons and classifier
        # elements in addition to the newly activated slices. Keep the
        # accumulated training delta outside serialized checkpoints so legacy
        # and adapter fingerprints remain stable, while deepcopy()/to() still
        # carry it with a live simulation model.
        for buffer_name, parameter in (
            ("_capacity_delta_first_weight", self.linear_relu_stack[0].weight),
            ("_capacity_delta_first_bias", self.linear_relu_stack[0].bias),
            ("_capacity_delta_second_weight", self.linear_relu_stack[2].weight),
            ("_capacity_delta_second_bias", self.linear_relu_stack[2].bias),
            ("_capacity_delta_classifier_weight", self.linear_relu_stack[4].weight),
            ("_capacity_delta_classifier_bias", self.linear_relu_stack[4].bias),
        ):
            self.register_buffer(
                buffer_name, torch.zeros_like(parameter), persistent=False,
            )

    @property
    def active_layer1_neurons(self) -> int:
        return int(torch.count_nonzero(self.layer1_mask).item())

    @property
    def active_layer2_neurons(self) -> int:
        return int(torch.count_nonzero(self.layer2_mask).item())

    def capacity_state(self) -> tuple[int, int]:
        return self.active_layer1_neurons, self.active_layer2_neurons

    def capacity_metrics(self) -> dict[str, float | int]:
        active_layer1, active_layer2 = self.capacity_state()
        return {
            "active_layer1": active_layer1,
            "active_layer2": active_layer2,
            "layer1_capacity_ratio": active_layer1 / self.max_layer1_neurons,
            "layer2_capacity_ratio": active_layer2 / self.max_layer2_neurons,
            "overall_capacity_ratio": (
                (active_layer1 + active_layer2)
                / (self.max_layer1_neurons + self.max_layer2_neurons)
            ),
        }

    def hidden_neuron_masks(self) -> dict[str, torch.Tensor]:
        """Expose masks under the same names used by neural-impact mapping."""
        return {
            "linear_relu_stack.0": self.layer1_mask,
            "linear_relu_stack.2": self.layer2_mask,
        }

    def _capacity_training_deltas(self) -> dict[str, torch.Tensor]:
        return {
            "linear_relu_stack.0.weight": self._capacity_delta_first_weight,
            "linear_relu_stack.0.bias": self._capacity_delta_first_bias,
            "linear_relu_stack.2.weight": self._capacity_delta_second_weight,
            "linear_relu_stack.2.bias": self._capacity_delta_second_bias,
            "linear_relu_stack.4.weight": self._capacity_delta_classifier_weight,
            "linear_relu_stack.4.bias": self._capacity_delta_classifier_bias,
        }

    @torch.no_grad()
    def accumulate_capacity_training_delta(
        self, parameter_deltas: Mapping[str, torch.Tensor],
    ) -> None:
        """Compose an accepted capacity-training update with future repairs."""
        buffers = self._capacity_training_deltas()
        if set(parameter_deltas) != set(buffers):
            raise ValueError("Capacity-training deltas must cover model parameters exactly")
        for name, buffer in buffers.items():
            delta = parameter_deltas[name]
            if delta.shape != buffer.shape or not torch.isfinite(delta).all():
                raise ValueError(f"Capacity-training delta for {name!r} is invalid")
            buffer.add_(delta.to(buffer))

    @torch.no_grad()
    def _zero_capacity_training_deltas(self) -> None:
        for delta in self._capacity_training_deltas().values():
            delta.zero_()

    @staticmethod
    def _layer_number(layer: int | str) -> int:
        aliases = {
            1: 1,
            2: 2,
            "layer1": 1,
            "layer2": 2,
            "linear_relu_stack.0": 1,
            "linear_relu_stack.2": 2,
        }
        try:
            return aliases[layer]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Unknown elastic hidden layer: {layer!r}") from exc

    def _layer_mask(self, layer: int | str) -> torch.Tensor:
        return self.layer1_mask if self._layer_number(layer) == 1 else self.layer2_mask

    def active_neuron_indices(self, layer: int | str) -> list[int]:
        mask = self._layer_mask(layer)
        return torch.nonzero(mask, as_tuple=False).flatten().cpu().tolist()

    def dormant_neuron_indices(self, layer: int | str) -> list[int]:
        mask = self._layer_mask(layer)
        return torch.nonzero(~mask, as_tuple=False).flatten().cpu().tolist()

    def validate_capacity_state(self) -> None:
        if self.layer1_mask.dtype != torch.bool or self.layer2_mask.dtype != torch.bool:
            raise ValueError("Elastic capacity masks must be boolean")
        if self.layer1_mask.shape != (self.max_layer1_neurons,):
            raise ValueError("Elastic layer-1 capacity mask has the wrong shape")
        if self.layer2_mask.shape != (self.max_layer2_neurons,):
            raise ValueError("Elastic layer-2 capacity mask has the wrong shape")
        if self.layer1_clone_sources.shape != (self.max_layer1_neurons,):
            raise ValueError("Elastic layer-1 clone-source state has the wrong shape")
        if self.layer2_clone_sources.shape != (self.max_layer2_neurons,):
            raise ValueError("Elastic layer-2 clone-source state has the wrong shape")
        for sources, maximum, minimum in (
            (self.layer1_clone_sources, self.max_layer1_neurons, self.min_layer1_neurons),
            (self.layer2_clone_sources, self.max_layer2_neurons, self.min_layer2_neurons),
        ):
            if (
                bool(torch.any(sources[:minimum] != -1))
                or bool(torch.any(sources < -1))
                or bool(torch.any(sources >= maximum))
            ):
                raise ValueError("Elastic clone-source state is invalid")
        active_layer1, active_layer2 = self.capacity_state()
        if not self.min_layer1_neurons <= active_layer1 <= self.max_layer1_neurons:
            raise ValueError("Elastic layer-1 capacity is outside its supported bounds")
        if not self.min_layer2_neurons <= active_layer2 <= self.max_layer2_neurons:
            raise ValueError("Elastic layer-2 capacity is outside its supported bounds")
        if not bool(torch.all(self.layer1_mask[:self.min_layer1_neurons])):
            raise ValueError("Elastic layer-1 minimum-prefix neurons must remain active")
        if not bool(torch.all(self.layer2_mask[:self.min_layer2_neurons])):
            raise ValueError("Elastic layer-2 minimum-prefix neurons must remain active")
        parameters = dict(self.named_parameters())
        for name, delta in self._capacity_training_deltas().items():
            if delta.shape != parameters[name].shape or not torch.isfinite(delta).all():
                raise ValueError("Elastic capacity-training delta state is invalid")

    def _hidden_activations(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer1 = self.linear_relu_stack[1](self.linear_relu_stack[0](x))
        layer1 = layer1.masked_fill(~self.layer1_mask, 0.0)
        layer2 = self.linear_relu_stack[3](self.linear_relu_stack[2](layer1))
        layer2 = layer2.masked_fill(~self.layer2_mask, 0.0)
        return layer1, layer2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, layer2 = self._hidden_activations(x)
        return self.linear_relu_stack[5](self.linear_relu_stack[4](layer2))

    @torch.inference_mode()
    def hidden_activation_magnitudes(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        layer1, layer2 = self._hidden_activations(x)
        return {
            "linear_relu_stack.0": layer1.abs().mean(dim=0),
            "linear_relu_stack.2": layer2.abs().mean(dim=0),
        }

    @staticmethod
    def _copy_with_deterministic_noise(
        target: torch.Tensor,
        source: torch.Tensor,
        noise: float,
        salt: int,
    ) -> None:
        values = source.detach().to(target)
        if noise == 0.0:
            target.copy_(values)
            return
        offsets = torch.arange(
            values.numel(), device=target.device, dtype=target.dtype,
        ).reshape(values.shape)
        signs = torch.where(
            (offsets.to(torch.int64) + salt) % 2 == 0,
            torch.ones_like(offsets),
            -torch.ones_like(offsets),
        )
        magnitudes = ((offsets + salt) % 7 + 1) / 7
        target.copy_(values + noise * signs * magnitudes)

    @torch.no_grad()
    def activate_neurons(
        self,
        layer: int | str,
        count: int,
        impact_scores: Sequence[float] | torch.Tensor,
        initialization_noise: float,
    ) -> list[int]:
        """Activate lowest-index dormant neurons by cloning the impact leader."""
        if count < 0:
            raise ValueError("Elastic expansion count must be nonnegative")
        if initialization_noise < 0:
            raise ValueError("Elastic initialization noise must be nonnegative")
        layer_number = self._layer_number(layer)
        mask = self._layer_mask(layer_number)
        scores = torch.as_tensor(
            impact_scores, dtype=torch.float32, device=mask.device,
        ).flatten()
        if scores.numel() != mask.numel() or not torch.isfinite(scores).all():
            raise ValueError("Elastic impact scores must be finite and match the layer width")
        active = self.active_neuron_indices(layer_number)
        dormant = self.dormant_neuron_indices(layer_number)
        if not active:
            raise ValueError("Elastic expansion requires an active source neuron")
        targets = dormant[:min(count, len(dormant))]
        if not targets:
            return []
        source = max(active, key=lambda index: (float(scores[index].item()), -index))

        first = self.linear_relu_stack[0]
        second = self.linear_relu_stack[2]
        classifier = self.linear_relu_stack[4]
        for target_index in targets:
            salt = 31 * layer_number + target_index
            if layer_number == 1:
                self._copy_with_deterministic_noise(
                    first.weight[target_index], first.weight[source],
                    initialization_noise, salt,
                )
                self._copy_with_deterministic_noise(
                    first.bias[target_index], first.bias[source],
                    initialization_noise, salt + 1,
                )
                active_layer2 = self.active_neuron_indices(2)
                # Split the source's outgoing contribution between source and
                # clone (Net2Net-style). With zero perturbation this preserves
                # the function while still initializing the associated edge.
                outgoing = second.weight[active_layer2, source].clone()
                second.weight[active_layer2, source] = 0.5 * outgoing
                second.weight[active_layer2, target_index] = 0.5 * outgoing
                self.layer1_clone_sources[target_index] = source
            else:
                self._copy_with_deterministic_noise(
                    second.weight[target_index], second.weight[source],
                    initialization_noise, salt,
                )
                self._copy_with_deterministic_noise(
                    second.bias[target_index], second.bias[source],
                    initialization_noise, salt + 1,
                )
                outgoing = classifier.weight[:, source].clone()
                classifier.weight[:, source].copy_(0.5 * outgoing)
                classifier.weight[:, target_index].copy_(0.5 * outgoing)
                self.layer2_clone_sources[target_index] = source
            mask[target_index] = True
        self.validate_capacity_state()
        return targets

    @torch.no_grad()
    def deactivate_neurons(
        self, layer: int | str, indices: Sequence[int] | torch.Tensor,
    ) -> list[int]:
        """Mask active neurons while retaining their parameters for rollback."""
        layer_number = self._layer_number(layer)
        mask = self._layer_mask(layer_number)
        requested: list[int] = []
        for raw_index in torch.as_tensor(indices).flatten().cpu().tolist():
            index = int(raw_index)
            if index not in requested:
                requested.append(index)
        if any(index < 0 or index >= mask.numel() for index in requested):
            raise ValueError("Elastic contraction contains an out-of-range neuron index")
        if any(not bool(mask[index].item()) for index in requested):
            raise ValueError("Elastic contraction can only deactivate active neurons")
        minimum = (
            self.min_layer1_neurons if layer_number == 1 else self.min_layer2_neurons
        )
        if any(index < minimum for index in requested):
            raise ValueError("Elastic contraction cannot deactivate minimum-prefix neurons")
        allowable = max(0, int(torch.count_nonzero(mask).item()) - minimum)
        deactivated = requested[:allowable]
        if deactivated:
            mask[deactivated] = False
        self.validate_capacity_state()
        return deactivated

    @torch.no_grad()
    def preserve_capacity_state_from(
        self, source: "ElasticTwoHiddenLayerModel",
    ) -> None:
        """Overlay dynamic capacity state on a fresh base-relative repair.

        Only parameter slices outside the dataset-specific minimum subnetwork
        are copied. Thus a newly predicted repair remains authoritative for
        minimum-prefix parameters while learned elastic rows, columns, and
        masks survive the repair anchor reset.
        """
        if type(source) is not type(self):
            raise TypeError("Elastic capacity state requires the same elastic model type")
        source.validate_capacity_state()
        first = self.linear_relu_stack[0]
        source_first = source.linear_relu_stack[0]
        second = self.linear_relu_stack[2]
        source_second = source.linear_relu_stack[2]
        classifier = self.linear_relu_stack[4]
        source_classifier = source.linear_relu_stack[4]

        first.weight[self.min_layer1_neurons:].copy_(
            source_first.weight[self.min_layer1_neurons:].to(first.weight)
        )
        first.bias[self.min_layer1_neurons:].copy_(
            source_first.bias[self.min_layer1_neurons:].to(first.bias)
        )
        second.weight[self.min_layer2_neurons:].copy_(
            source_second.weight[self.min_layer2_neurons:].to(second.weight)
        )
        second.bias[self.min_layer2_neurons:].copy_(
            source_second.bias[self.min_layer2_neurons:].to(second.bias)
        )
        second.weight[:, self.min_layer1_neurons:].copy_(
            source_second.weight[:, self.min_layer1_neurons:].to(second.weight)
        )
        classifier.weight[:, self.min_layer2_neurons:].copy_(
            source_classifier.weight[:, self.min_layer2_neurons:].to(classifier.weight)
        )
        layer1_sources = torch.unique(
            source.layer1_clone_sources[
                source.layer1_mask & (source.layer1_clone_sources >= 0)
            ]
        ).tolist()
        layer2_sources = torch.unique(
            source.layer2_clone_sources[
                source.layer2_mask & (source.layer2_clone_sources >= 0)
            ]
        ).tolist()
        if layer1_sources:
            second.weight[:, layer1_sources] = source_second.weight[
                :, layer1_sources
            ].to(second.weight)
        if layer2_sources:
            classifier.weight[:, layer2_sources] = source_classifier.weight[
                :, layer2_sources
            ].to(classifier.weight)

        # Exact-copied elastic slices already include their capacity-training
        # updates. For the remaining legacy-active elements, add the cumulative
        # training delta to the new base-relative repair so later drift repairs
        # do not silently erase accepted capacity learning.
        parameters = dict(self.named_parameters())
        copied = {
            name: torch.zeros_like(parameter, dtype=torch.bool)
            for name, parameter in parameters.items()
        }
        copied["linear_relu_stack.0.weight"][self.min_layer1_neurons:] = True
        copied["linear_relu_stack.0.bias"][self.min_layer1_neurons:] = True
        copied["linear_relu_stack.2.weight"][self.min_layer2_neurons:] = True
        copied["linear_relu_stack.2.weight"][:, self.min_layer1_neurons:] = True
        copied["linear_relu_stack.2.bias"][self.min_layer2_neurons:] = True
        copied["linear_relu_stack.4.weight"][:, self.min_layer2_neurons:] = True
        if layer1_sources:
            copied["linear_relu_stack.2.weight"][:, layer1_sources] = True
        if layer2_sources:
            copied["linear_relu_stack.4.weight"][:, layer2_sources] = True
        source_deltas = source._capacity_training_deltas()
        target_deltas = self._capacity_training_deltas()
        for name, parameter in parameters.items():
            delta = source_deltas[name].to(parameter)
            retained = ~copied[name]
            parameter[retained] = parameter[retained] + delta[retained]
            target_deltas[name].copy_(source_deltas[name].to(delta))
        self.layer1_mask.copy_(source.layer1_mask.to(self.layer1_mask))
        self.layer2_mask.copy_(source.layer2_mask.to(self.layer2_mask))
        self.layer1_clone_sources.copy_(
            source.layer1_clone_sources.to(self.layer1_clone_sources)
        )
        self.layer2_clone_sources.copy_(
            source.layer2_clone_sources.to(self.layer2_clone_sources)
        )
        self.validate_capacity_state()

    @torch.no_grad()
    def load_compatible_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Strict-load an elastic state or embed the exact legacy schema."""
        keys = set(state)
        if self.legacy_checkpoint_prefix != "linear_relu_stack":
            legacy_keys = {
                name.replace(
                    "linear_relu_stack", self.legacy_checkpoint_prefix, 1,
                )
                for name in self._LEGACY_SHAPES
            }
            if keys == legacy_keys:
                state = {
                    name.replace(
                        self.legacy_checkpoint_prefix, "linear_relu_stack", 1,
                    ): value
                    for name, value in state.items()
                }
                keys = set(state)
        legacy_keys = set(self._LEGACY_SHAPES)
        if keys == legacy_keys:
            if any(
                not isinstance(state[name], torch.Tensor)
                or tuple(state[name].shape) != shape
                for name, shape in self._LEGACY_SHAPES.items()
            ):
                raise ValueError("Legacy checkpoint tensor shapes are incompatible")
            for parameter in self.parameters():
                parameter.zero_()
            first = self.linear_relu_stack[0]
            second = self.linear_relu_stack[2]
            classifier = self.linear_relu_stack[4]
            first.weight[:self.min_layer1_neurons].copy_(
                state["linear_relu_stack.0.weight"].to(first.weight)
            )
            first.bias[:self.min_layer1_neurons].copy_(
                state["linear_relu_stack.0.bias"].to(first.bias)
            )
            second.weight[:self.min_layer2_neurons, :self.min_layer1_neurons].copy_(
                state["linear_relu_stack.2.weight"].to(second.weight)
            )
            second.bias[:self.min_layer2_neurons].copy_(
                state["linear_relu_stack.2.bias"].to(second.bias)
            )
            classifier.weight[:, :self.min_layer2_neurons].copy_(
                state["linear_relu_stack.4.weight"].to(classifier.weight)
            )
            classifier.bias.copy_(state["linear_relu_stack.4.bias"].to(classifier.bias))
            self.layer1_mask.zero_()
            self.layer1_mask[:self.min_layer1_neurons] = True
            self.layer2_mask.zero_()
            self.layer2_mask[:self.min_layer2_neurons] = True
            self.layer1_clone_sources.fill_(-1)
            self.layer2_clone_sources.fill_(-1)
            self._zero_capacity_training_deltas()
            self.validate_capacity_state()
            return

        expected_state = self.state_dict()
        if keys != set(expected_state) or any(
            not isinstance(state[name], torch.Tensor)
            or state[name].shape != expected_state[name].shape
            for name in expected_state
        ):
            raise ValueError(
                "Elastic checkpoint must use either the exact legacy or elastic schema"
            )
        self.load_state_dict(state, strict=True)
        self._zero_capacity_training_deltas()
        self.validate_capacity_state()


class ID_CIC_IDS2017_elastic_pforest(ElasticTwoHiddenLayerModel):
    """Elastic CIC student preserving the legacy ``16 -> 8 -> 4 -> 2`` prefix."""


class ID_CIC_IDS2017_large_pforest(InferenceModel):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        sizes = [16, 128, 64, 32, 16, 8, 4, 2]
        for index, (left, right) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(left, right))
            layers.append(nn.Softmax(dim=1) if index == len(sizes) - 2 else nn.ReLU())
        self.linear_relu_stack = nn.Sequential(*layers)
        self.linear_relu_stack.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_relu_stack(x)


class ID_UNSW_NB15_N3IC(InferenceModel):
    def __init__(self, input_shape: int = 20, neurons: list[int] | None = None) -> None:
        super().__init__()
        neurons = neurons or [32, 16, 2]
        self.model = _two_hidden_stack(input_shape, neurons[0], neurons[1], neurons[2])
        self.model.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNSW_IoT_N3IC(InferenceModel):
    def __init__(self, input_shape: int = 16, neurons: list[int] | None = None) -> None:
        super().__init__()
        neurons = neurons or [64, 32, 10]
        self.model = _two_hidden_stack(input_shape, neurons[0], neurons[1], neurons[2])
        self.model.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ID_UNSW_NB15_elastic_N3IC(ElasticTwoHiddenLayerModel):
    """Elastic UNSW-NB15 student preserving the legacy ``20 -> 32 -> 16 -> 2`` prefix."""

    min_layer1_neurons = 32
    min_layer2_neurons = 16
    max_layer1_neurons = 64
    max_layer2_neurons = 32
    input_features = 20
    output_classes = 2
    legacy_checkpoint_prefix = "model"
    _LEGACY_SHAPES = {
        "linear_relu_stack.0.weight": (32, 20),
        "linear_relu_stack.0.bias": (32,),
        "linear_relu_stack.2.weight": (16, 32),
        "linear_relu_stack.2.bias": (16,),
        "linear_relu_stack.4.weight": (2, 16),
        "linear_relu_stack.4.bias": (2,),
    }

    def __init__(
        self, input_shape: int = 20, neurons: list[int] | None = None,
    ) -> None:
        expected = [self.min_layer1_neurons, self.min_layer2_neurons, self.output_classes]
        if input_shape != self.input_features or (neurons is not None and neurons != expected):
            raise ValueError(
                "Elastic UNSW-NB15 model requires input_shape=20 and neurons=[32, 16, 2]"
            )
        super().__init__()

class UNSW_IoT_elastic_N3IC(ElasticTwoHiddenLayerModel):
    """Elastic UNSW-IoT student preserving the legacy ``16 -> 64 -> 32 -> 10`` prefix."""

    min_layer1_neurons = 64
    min_layer2_neurons = 32
    max_layer1_neurons = 128
    max_layer2_neurons = 64
    input_features = 16
    output_classes = 10
    legacy_checkpoint_prefix = "model"
    _LEGACY_SHAPES = {
        "linear_relu_stack.0.weight": (64, 16),
        "linear_relu_stack.0.bias": (64,),
        "linear_relu_stack.2.weight": (32, 64),
        "linear_relu_stack.2.bias": (32,),
        "linear_relu_stack.4.weight": (10, 32),
        "linear_relu_stack.4.bias": (10,),
    }

    def __init__(
        self, input_shape: int = 16, neurons: list[int] | None = None,
    ) -> None:
        expected = [self.min_layer1_neurons, self.min_layer2_neurons, self.output_classes]
        if input_shape != self.input_features or (neurons is not None and neurons != expected):
            raise ValueError(
                "Elastic UNSW-IoT model requires input_shape=16 and neurons=[64, 32, 10]"
            )
        super().__init__()

MODEL_REGISTRY = {
    cls.__name__: cls for cls in (
        ID_CIC_IDS2017_small_pforest,
        ID_CIC_IDS2017_elastic_pforest,
        ID_CIC_IDS2017_large_pforest,
        ID_UNSW_NB15_N3IC,
        ID_UNSW_NB15_elastic_N3IC,
        UNSW_IoT_N3IC,
        UNSW_IoT_elastic_N3IC,
    )
}


def build_model(name: str, kwargs: dict[str, Any] | None = None) -> InferenceModel:
    try:
        return MODEL_REGISTRY[name](**(kwargs or {}))
    except KeyError as exc:
        raise ValueError(f"Unsupported DRIFTADAPT model: {name}") from exc


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> None:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        state = torch.load(path, map_location=device)
    if isinstance(model, ElasticTwoHiddenLayerModel):
        model.load_compatible_state_dict(state)
    else:
        model.load_state_dict(state)
