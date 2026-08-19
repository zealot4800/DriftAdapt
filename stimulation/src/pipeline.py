from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import torch

from .data import StreamingDataset, resolve_path
from .labelers import build_labeler
from .metrics import macro_f1, mean
from .models import build_model, load_checkpoint
from .trainer import form_training_set, retrain_model
from .trigger import RetrainingTrigger


class DriftAdaptPipeline:
    def __init__(self, config: dict, config_path: Path) -> None:
        self.config = config
        runtime = config.get("runtime", {})
        requested = runtime.get("device", "cuda")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        torch.manual_seed(int(runtime.get("seed", 42)))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(runtime.get("seed", 42)))

        base = config_path.parent
        dataset_config = copy.deepcopy(config["dataset"])
        dataset_config["files"] = [str(resolve_path(path, base)) for path in dataset_config["files"]]
        self.dataset = StreamingDataset(dataset_config, base, self.device)

        model_config = config["model"]
        self.model = build_model(model_config["class"], model_config.get("kwargs")).to(self.device)
        checkpoint = resolve_path(model_config["checkpoint"], base)
        load_checkpoint(self.model, str(checkpoint), self.device)
        self.model.eval()
        self.static_model = copy.deepcopy(self.model).eval()
        self.oracle_model = copy.deepcopy(self.model).eval() if config.get("oracle", {}).get("enabled") else None

        labeler_config = copy.deepcopy(config["labeler"])
        labeler_config.setdefault("feature_names", dataset_config["feature_names"])
        if labeler_config.get("checkpoint"):
            labeler_config["checkpoint"] = str(resolve_path(labeler_config["checkpoint"], base))
        self.labeler = build_labeler(labeler_config, self.device)
        self.optimizer = None
        trigger_config = config["trigger"]
        if trigger_config.get("type") != "accuracy_proxy":
            raise ValueError("This drift-only implementation requires trigger.type: accuracy_proxy")
        self.trigger = RetrainingTrigger(threshold=float(trigger_config["threshold"]))
        self.base_dir = base

    @torch.inference_mode()
    def _predict(self, model, features):
        return model.run_inference(features)[1]

    def run(self) -> tuple[list[dict], dict]:
        rows: list[dict] = []
        stream = self.config["stream"]
        training = self.config["training"]
        labeler_type = self.config["labeler"]["type"].lower()

        for index, window in enumerate(self.dataset.windows(int(stream["window_size"])), start=1):
            student_labels = self._predict(self.model, window.features)
            static_labels = self._predict(self.static_model, window.features)
            oracle_labels = self._predict(self.oracle_model, window.features) if self.oracle_model else None
            produced = self.labeler.label(window.features, window.raw_features, window.generated_labels)

            if labeler_type == "device_list":
                valid = produced != -1
                # The artifact uses publisher labels for flows identified by the
                # device list, while the generated values let us report label quality.
                labeler_f1 = macro_f1(window.labels[valid], produced[valid])
                labeler_labels = window.labels[valid]
                training_features = window.features[valid]
                proxy_student = student_labels[valid]
                proxy_static = static_labels[valid]
            else:
                labeler_labels = produced
                training_features = window.features
                proxy_student = student_labels
                proxy_static = static_labels
                labeler_f1 = macro_f1(window.labels, labeler_labels)

            proxy_f1 = macro_f1(labeler_labels, proxy_student)
            static_proxy_f1 = macro_f1(labeler_labels, proxy_static)
            requested = self.trigger.update(
                window_index=index, model_proxy_f1=proxy_f1, static_proxy_f1=static_proxy_f1,
            )
            balanced = bool(training.get("balance_binary", self.config["dataset"]["num_classes"] == 2))
            train_x, train_y = form_training_set(training_features, labeler_labels, balance_binary=balanced)
            retrained = requested and train_x is not None
            training_time = 0.0
            sample_count = int(train_y.numel()) if retrained else 0
            if retrained:
                training_time, self.optimizer = retrain_model(
                    self.model, train_x, train_y,
                    epochs=int(training["epochs"]), batch_size=int(training["batch_size"]),
                    learning_rate=float(training["learning_rate"]),
                    optimizer_name=training.get("optimizer", "Adam"), optimizer=self.optimizer,
                )

            row = {
                "window": index,
                "student_f1": macro_f1(window.labels, student_labels),
                "static_f1": macro_f1(window.labels, static_labels),
                "labeler_f1": labeler_f1,
                "proxy_f1": proxy_f1,
                "static_proxy_f1": static_proxy_f1,
                "retrained": retrained,
                "training_samples": sample_count,
                "training_time": training_time,
            }
            if oracle_labels is not None:
                row["oracle_f1"] = macro_f1(window.labels, oracle_labels)
            rows.append(row)

        summary = self._save(rows)
        return rows, summary

    def _save(self, rows: list[dict]) -> dict:
        dataset_name = self.config["dataset"]["name"]
        output = Path(self.config.get("output", {}).get("directory", "results"))
        if not output.is_absolute():
            output = (self.base_dir.parent / output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(output / f"{dataset_name}.csv", index=False)
        times = [float(row["training_time"]) for row in rows if row["retrained"]]
        summary = {
            "dataset": dataset_name,
            "number_of_windows": len(rows),
            "number_of_retraining_events": len(times),
            "average_student_f1": mean([row["student_f1"] for row in rows]),
            "average_static_f1": mean([row["static_f1"] for row in rows]),
            "average_labeler_f1": mean([row["labeler_f1"] for row in rows]),
            "total_retraining_time": sum(times),
            "average_retraining_time": mean(times),
        }
        with (output / f"{dataset_name}.json").open("w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "windows": rows}, handle, indent=2, allow_nan=True)
        return summary
