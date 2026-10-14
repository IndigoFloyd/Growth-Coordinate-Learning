"""Reusable source-statistics alignment; no dataset-specific training branches."""

import ctypes
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn

from .analysis import compute_hidden_features
from .dataloader import make_loader
from .models import (
    LowRankAdapter,
    MLP_MultiOutput,
    MLP_MultiOutputWithAdapter,
    MLPWithAdapter,
    load_model,
    set_global_seed,
)


@contextmanager
def source_solver(shuffled_batches):
    """Keep the published shuffled-batch eigensolver profile local to source statistics."""
    if not shuffled_batches:
        yield
        return
    mkl = ctypes.CDLL(str(Path(torch.__file__).parent / "lib/libtorch_cpu.so"))
    get_dynamic = mkl.mkl_serv_get_dynamic
    get_dynamic.argtypes, get_dynamic.restype = [], ctypes.c_int
    set_dynamic = mkl.MKL_Set_Dynamic
    set_dynamic.argtypes, set_dynamic.restype = [ctypes.c_int], None
    set_local = mkl.MKL_Set_Num_Threads_Local
    set_local.argtypes, set_local.restype = [ctypes.c_int], ctypes.c_int
    previous_dynamic = get_dynamic()
    previous_local = set_local(2)
    set_dynamic(0)
    try:
        yield
    finally:
        set_local(previous_local)
        set_dynamic(previous_dynamic)


def turn_off_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            module.eval()


def compute_source_statistics(
    source_loader,
    model: nn.Module,
    device: torch.device,
    top_k: int = 10,
    weight_bias: float = 1.0,
    weight_exp: float = 1.0,
    var_epsilon: float = 1e-5,
    normalize_alpha: bool = True,
) -> Dict[str, torch.Tensor]:
    model.eval()
    all_features = []

    with torch.no_grad():
        for batch in source_loader:
            inputs = batch["latent"].to(device)
            features = model.forward_feature(inputs)
            all_features.append(features.detach().cpu())

    if not all_features:
        raise ValueError("Source loader returned no features.")
    all_features = torch.cat(all_features, dim=0)
    n_samples, dim = all_features.shape
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to compute source statistics.")

    mu_s = all_features.mean(dim=0)
    centered = all_features - mu_s
    cov = (centered.T @ centered) / (n_samples - 1)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    idx = torch.argsort(eigenvalues, descending=True)
    top_k = min(top_k, dim)
    idx = idx[:top_k]

    lambda_s = eigenvalues[idx].clamp(min=var_epsilon)
    V_s = eigenvectors[:, idx].T

    base_model = getattr(model, "base", model)
    if isinstance(base_model, MLP_MultiOutput):
        head_ws = [head.weight.detach().cpu() for head in base_model.output_heads]
        # Multi-task aggregation: use mean absolute head weight instead of max
        # to avoid over-dominance by a single task head.
        w = torch.stack(head_ws).abs().mean(dim=0).squeeze(0)
    else:
        w = base_model.head.weight.detach().cpu().squeeze(0)

    proj = torch.abs(V_s @ w)
    alpha = (proj + weight_bias).pow(weight_exp)
    if normalize_alpha:
        alpha = alpha / (alpha.sum() + 1e-12)

    return {
        "mu_s": mu_s.to(device),
        "V_s": V_s.to(device),
        "source_var": lambda_s.to(device),
        "alpha": alpha.to(device),
    }


def diagonal_gaussian_kl_loss(
    m1: torch.Tensor,
    v1: torch.Tensor,
    m2: torch.Tensor,
    v2: torch.Tensor,
    eps: float = 0.0,
    dim_reduction: str = "sum",
) -> torch.Tensor:
    loss = (v2.log() - v1.log() + (v1 + (m2 - m1).square()) / (v2 + eps) - 1) / 2
    if dim_reduction == "sum":
        return loss.sum()
    if dim_reduction == "mean":
        return loss.mean()
    if dim_reduction == "none":
        return loss
    raise ValueError(f"Invalid dim_reduction: {dim_reduction!r}")


def compute_ssa_loss_from_pc(
    f_pc: torch.Tensor,
    source_var: torch.Tensor,
    alpha: torch.Tensor,
    loss_config: dict,
    eps: float = 1e-5,
) -> torch.Tensor:
    f_mean = f_pc.mean(dim=0)
    if f_pc.shape[0] < 2:
        f_var = torch.zeros_like(f_mean) + eps
    else:
        f_var = f_pc.var(dim=0, unbiased=False).clamp(min=eps)

    zeros = torch.zeros_like(f_mean)
    loss_cfg = dict(loss_config)
    loss_cfg.setdefault("dim_reduction", "none")

    kl_dist = diagonal_gaussian_kl_loss(
        f_mean, f_var, zeros, source_var, **loss_cfg
    ) + diagonal_gaussian_kl_loss(zeros, source_var, f_mean, f_var, **loss_cfg)
    return kl_dist @ alpha


def ssa_loss(
    target_features: torch.Tensor,
    source_stats: Dict[str, torch.Tensor],
    loss_config: dict,
    eps: float = 1e-5,
) -> torch.Tensor:
    mu_s = source_stats["mu_s"]
    V_s = source_stats["V_s"]
    source_var = source_stats["source_var"]
    alpha = source_stats["alpha"]

    z_centered = target_features - mu_s
    f_pc = z_centered @ V_s.T
    return compute_ssa_loss_from_pc(f_pc, source_var, alpha, loss_config, eps=eps)


def train_ssa(
    source_features,
    target_features,
    source_weights,
    *,
    epochs=20,
    lr=0.005,
    rank=32,
    top_k=70,
    seed=42,
    batch_size=256,
    mode="ln_lora",
    shuffle=False,
    reference="source",
    device="cuda",
    step_callback=None,
):
    """Fit the same SSA objective for any source/target feature matrices."""
    set_global_seed(seed)
    device = torch.device(device)
    base, source_payload = load_model(source_weights, device)
    adapter = LowRankAdapter(base.fc1.in_features, rank=rank)
    model = (MLP_MultiOutputWithAdapter if isinstance(base, MLP_MultiOutput) else MLPWithAdapter)(
        base, adapter
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode in ["lora", "ln_lora"]:
        for parameter in model.adapter.parameters():
            parameter.requires_grad = True
    if mode in ["ln", "ln_lora"]:
        for parameter in model.base.ln.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    source_loader = make_loader(source_features, batch_size)
    # The manuscript uses the same shuffled batch order across adaptation seeds.
    target_loader = make_loader(target_features, batch_size, shuffle, seed=42)
    with source_solver(shuffle):
        statistics = compute_source_statistics(source_loader, model, device, top_k=top_k)
    if reference != "source":
        initial = torch.from_numpy(
            compute_hidden_features(model, target_features, batch_size=1024, device=device)
        ).to(device)
        projected = (initial - statistics["mu_s"]) @ statistics["V_s"].T
        if reference in ["target_variance", "target_both"]:
            statistics["source_var"] = projected.var(dim=0, unbiased=True).clamp(min=1e-5)
        if reference in ["target_mean", "target_both"]:
            statistics["mu_s"] = initial.mean(dim=0)
    if step_callback is not None:
        step_callback(model, 0, statistics, None)
    model.train()
    turn_off_dropout(model)
    step, history = 0, []
    for epoch in range(epochs):
        losses = []
        for batch in target_loader:
            features = model.forward_feature(batch["latent"].to(device))
            loss = ssa_loss(features, statistics, loss_config={})
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            losses.append(float(loss.detach().item()))
            if step_callback is not None:
                step_callback(model, step, statistics, losses[-1])
                model.train()
                turn_off_dropout(model)
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses))))
    model.eval()
    return model, dict(
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        target_names=source_payload.get("target_names", source_payload.get("targets")),
        source_weights=str(source_weights),
        seed=seed,
        rank=rank,
        top_k=top_k,
        mode=mode,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        shuffle=shuffle,
        shuffle_seed=42 if shuffle else None,
        reference=reference,
        history=history,
    )
