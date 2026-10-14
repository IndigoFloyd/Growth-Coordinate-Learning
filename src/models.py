from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


def scalar_head(head, inputs):
    # Column-major storage avoids the cuBLAS reduction change for short scalar-head batches.
    weight = nn.functional.pad(head.weight.T, (0, 1)).T[:1]
    return nn.functional.linear(inputs, weight, head.bias)


class LowRankAdapter(nn.Module):
    """Residual low-rank adapter used by SSA-tuned projectors."""

    def __init__(self, dim: int, rank: int = 32, scale: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.scale = nn.Parameter(torch.tensor(scale))
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.GELU = nn.GELU()
        self.LN = nn.LayerNorm(dim)

        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.GELU(self.B(self.drop(self.GELU(self.A(x)))))


class MLP(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fc1(x)
        z = self.ln(z)
        z = self.act(z)
        return z

    def forward_feature_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_feature(x)
        z = self.drop(z)
        y = scalar_head(self.head, z)
        return y.squeeze(-1)


class MLP_MultiOutput(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, target_names=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.output_metrics = (
            list(target_names)
            if target_names is not None
            else ["FreshWeightShoot", "DryWeightShoot", "Height", "Diameter", "LeafArea"]
        )
        self.output_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in self.output_metrics])

    def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fc1(x)
        z = self.ln(z)
        z = self.act(z)
        return z

    def forward_feature_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.forward_feature(x)
        z = self.drop(z)
        outputs = {}
        for i, head in enumerate(self.output_heads):
            metric = self.output_metrics[i]
            outputs[metric] = scalar_head(head, z).squeeze(-1)
        return outputs


class MLPWithAdapter(nn.Module):
    def __init__(self, base: MLP, adapter: LowRankAdapter):
        super().__init__()
        self.base = base
        self.adapter = adapter

    def load_base_state_dict(self, state_dict: dict) -> None:
        self.base.load_state_dict(state_dict, strict=True)

    def forward_feature_raw(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        return self.base.fc1(x)

    def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_feature_raw(x)
        z = self.base.ln(z)
        z = self.base.act(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_feature(x)
        z = self.base.drop(z)
        y = scalar_head(self.base.head, z)
        return y.squeeze(-1)


class MLP_MultiOutputWithAdapter(nn.Module):
    def __init__(self, base: MLP_MultiOutput, adapter: LowRankAdapter):
        super().__init__()
        self.base = base
        self.adapter = adapter

    @property
    def output_metrics(self):
        return self.base.output_metrics

    def load_base_state_dict(self, state_dict: dict) -> None:
        self.base.load_state_dict(state_dict, strict=True)

    def forward_feature_raw(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        return self.base.fc1(x)

    def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_feature_raw(x)
        z = self.base.ln(z)
        z = self.base.act(z)
        return z

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.forward_feature(x)
        z = self.base.drop(z)
        outputs = {}
        for i, head in enumerate(self.base.output_heads):
            metric = self.base.output_metrics[i]
            outputs[metric] = scalar_head(head, z).squeeze(-1)
        return outputs


def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    # Required by CUDA>=10.2 when deterministic algorithms are enabled.
    # Keep externally provided value if user already configured it.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


ROOT = Path(__file__).resolve().parents[1]


def load_model(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    adapted = "base.fc1.weight" in state
    prefix = "base." if adapted else ""
    hidden, inputs = state[prefix + "fc1.weight"].shape
    multi = prefix + "output_heads.0.weight" in state
    targets = payload.get("target_names", payload.get("targets"))
    base = MLP_MultiOutput(inputs, hidden, targets) if multi else MLP(inputs, hidden)
    if adapted:
        adapter = LowRankAdapter(inputs, rank=state["adapter.A.weight"].shape[0])
        model = (MLP_MultiOutputWithAdapter if multi else MLPWithAdapter)(base, adapter)
    else:
        model = base
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload


def model_paths(config=ROOT / "data/models.yml"):
    values = yaml.safe_load(Path(config).read_text())
    return {
        dataset: {key: (ROOT / Path(path)).resolve() for key, path in models.items()}
        for dataset, models in values.items()
    }


def save_checkpoint(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
