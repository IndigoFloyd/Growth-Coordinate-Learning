"""Reusable supervised projector training for arbitrary numeric targets."""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from .dataloader import FeatureDataset
from .models import MLP, MLP_MultiOutput, set_global_seed


def forward_outputs(model, inputs, targets):
    if isinstance(model, MLP_MultiOutput):
        values = model(inputs)
        return torch.stack([values[name] for name in targets], dim=1)
    return model(inputs).unsqueeze(-1)


def run_epoch(model, loader, optimizer, device, targets, stats):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    for batch in loader:
        inputs = batch["latent"].to(device)
        truth = (batch["target"].to(device) - stats[0]) / stats[1]
        prediction = forward_outputs(model, inputs, targets)
        loss = nn.functional.mse_loss(prediction, truth)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * len(inputs)
        count += len(inputs)
    return total / count


def train_mlp(
    features,
    labels,
    targets,
    *,
    hidden_dim=256,
    epochs=150,
    updates=None,
    batch_size=128,
    lr=0.001,
    weight_decay=0.01,
    val_split=0.2,
    seed=42,
    train_indices=None,
    validation_indices=None,
    device="cuda",
):
    """Train one model. Optional fixed-update budget is used by the scaling experiment."""
    set_global_seed(seed)
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32).reshape(len(features), len(targets))
    if train_indices is None:
        indices = torch.randperm(len(features), generator=torch.Generator().manual_seed(seed)).tolist()
        n_val = max(1, int(len(features) * val_split))
        train_indices, validation_indices = indices[:-n_val], indices[-n_val:]
    train, val = list(train_indices), list(validation_indices)
    # Preserve the existing epoch-based multi-head initialization sequence.
    if len(targets) > 1 and updates is None:
        MLP_MultiOutput()
    model = (
        MLP(features.shape[1], hidden_dim)
        if len(targets) == 1
        else MLP_MultiOutput(features.shape[1], hidden_dim, targets)
    ).to(device)
    data = FeatureDataset(features, pd.DataFrame(labels, columns=targets), targets)
    training = torch.utils.data.Subset(data, train)
    if updates is None:
        train_loader = DataLoader(training, batch_size=batch_size, shuffle=True)
        rounds, stride = epochs, 1
    else:
        if updates % 2:
            raise ValueError("The fixed-update protocol checks validation every two updates.")
        sampler = RandomSampler(
            training,
            replacement=True,
            num_samples=2 * batch_size,
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(training, batch_size=batch_size, sampler=sampler)
        rounds, stride = updates // 2, 2
    val_loader = DataLoader(torch.utils.data.Subset(data, val), batch_size=batch_size)
    mean, std = labels[train].mean(axis=0), labels[train].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    stats = (torch.tensor(mean, device=device), torch.tensor(std, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_step, best_state, history = float("inf"), 0, None, []
    for iteration in range(1, rounds + 1):
        loss = run_epoch(model, train_loader, optimizer, device, targets, stats)
        validation = run_epoch(model, val_loader, None, device, targets, stats)
        step = iteration * stride
        history.append(dict(step=step, train_normalized_mse=loss, validation_normalized_mse=validation))
        if validation < best:
            best, best_step = validation, step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    payload = dict(
        state_dict=best_state,
        input_dim=features.shape[1],
        hidden_dim=hidden_dim,
        mode="single" if len(targets) == 1 else "multi",
        target_names=list(targets),
        target_mean=mean,
        target_std=std,
        seed=seed,
        best_step=best_step,
        best_validation_loss=best,
        history=history,
        train_indices=train,
        validation_indices=val,
        training=dict(
            epochs=epochs, updates=updates, batch_size=batch_size, lr=lr, weight_decay=weight_decay
        ),
    )
    return model, payload
