import json
import os
from copy import deepcopy
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import summarize_from_logits


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        output_dir: str,
        scheduler=None,
        monitor_metric: str = "qwk",
        monitor_mode: str = "max",
        early_stopping_patience: int = 5,
        n_bins: int = 15,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.output_dir = output_dir
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.early_stopping_patience = early_stopping_patience
        self.n_bins = n_bins

        os.makedirs(self.output_dir, exist_ok=True)

        self.best_model_state = None
        self.best_metric_value = None
        self.history: List[Dict] = []

    def _is_improved(self, current_value: float) -> bool:
        if self.best_metric_value is None:
            return True

        if self.monitor_mode == "max":
            return current_value > self.best_metric_value
        if self.monitor_mode == "min":
            return current_value < self.best_metric_value

        raise ValueError("monitor_mode must be either 'max' or 'min'")

    def _save_checkpoint(self, filename: str = "best_model.pt") -> None:
        save_path = os.path.join(self.output_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "best_metric_value": self.best_metric_value,
                "history": self.history,
            },
            save_path,
        )

    def _save_history(self) -> None:
        history_path = os.path.join(self.output_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def train_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        all_logits = []
        all_labels = []

        for batch in tqdm(loader, desc="Training", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = summarize_from_logits(all_logits, all_labels, n_bins=self.n_bins)
        metrics["loss"] = total_loss / len(loader.dataset)
        return metrics

    @torch.no_grad()
    def validate_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        all_logits = []
        all_labels = []

        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = summarize_from_logits(all_logits, all_labels, n_bins=self.n_bins)
        metrics["loss"] = total_loss / len(loader.dataset)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> List[Dict]:
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch [{epoch}/{num_epochs}]")

            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.validate_one_epoch(val_loader)

            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            self.history.append(epoch_record)

            current_metric = val_metrics[self.monitor_metric]

            print("Train Metrics:")
            for key, value in train_metrics.items():
                print(f"  {key}: {value:.4f}")

            print("Val Metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")

            if self._is_improved(current_metric):
                self.best_metric_value = current_metric
                self.best_model_state = deepcopy(self.model.state_dict())
                self._save_checkpoint("best_model.pt")
                patience_counter = 0
                print(f"Best model updated based on val_{self.monitor_metric}: {current_metric:.4f}")
            else:
                patience_counter += 1
                print(
                    f"No improvement. Early stopping patience: "
                    f"{patience_counter}/{self.early_stopping_patience}"
                )

            if self.scheduler is not None:
                try:
                    self.scheduler.step(val_metrics["loss"])
                except TypeError:
                    self.scheduler.step()

            self._save_history()

            if patience_counter >= self.early_stopping_patience:
                print("Early stopping triggered.")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    @torch.no_grad()
    def collect_logits(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        self.model.eval()

        all_logits = []
        all_labels = []
        all_ids = []

        for batch in tqdm(loader, desc="Collecting logits", leave=False):
            images = batch["image"].to(self.device)
            logits = self.model(images)

            all_logits.append(logits.detach().cpu())

            if "label" in batch:
                all_labels.append(batch["label"].detach().cpu())

            if "id" in batch:
                all_ids.extend(batch["id"])

        result = {
            "logits": torch.cat(all_logits, dim=0),
            "ids": all_ids,
        }

        if len(all_labels) > 0:
            result["labels"] = torch.cat(all_labels, dim=0)

        return result

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        self.model.eval()

        all_logits = []
        all_ids = []
        all_labels = []

        for batch in tqdm(loader, desc="Predicting", leave=False):
            images = batch["image"].to(self.device)
            logits = self.model(images)

            all_logits.append(logits.detach().cpu())
            all_ids.extend(batch["id"])

            if "label" in batch:
                all_labels.append(batch["label"].detach().cpu())

        outputs = {
            "logits": torch.cat(all_logits, dim=0),
            "ids": all_ids,
        }

        if len(all_labels) > 0:
            outputs["labels"] = torch.cat(all_labels, dim=0)

        return outputs