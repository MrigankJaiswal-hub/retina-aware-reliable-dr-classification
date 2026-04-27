from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling module.
    Calibrated logits = logits / T
    """

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.clamp(self.temperature, min=1e-6)
        return logits / temperature

    def get_temperature(self) -> float:
        return float(torch.clamp(self.temperature, min=1e-6).item())


def fit_temperature_scaler(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.01,
    device: str = "cpu",
) -> Tuple[TemperatureScaler, Dict[str, float]]:
    """
    Fit temperature scaling on validation logits by minimizing NLL.
    """
    logits = logits.to(device)
    labels = labels.to(device)

    scaler = TemperatureScaler().to(device)

    before_nll = F.cross_entropy(logits, labels).item()

    optimizer = optim.LBFGS(
        scaler.parameters(),
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        scaled_logits = scaler(logits)
        loss = F.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        scaled_logits = scaler(logits)
        after_nll = F.cross_entropy(scaled_logits, labels).item()

    result = {
        "temperature": scaler.get_temperature(),
        "nll_before": float(before_nll),
        "nll_after": float(after_nll),
    }

    return scaler, result


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    return logits / temperature