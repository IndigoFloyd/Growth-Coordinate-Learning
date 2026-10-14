"""Train a supervised MLP on any configured dataset and numeric target columns."""

import argparse
import json
from pathlib import Path

import torch

from src.analysis import regression_metrics
from src.dataloader import CONFIG, load_dataset
from src.mlp import forward_outputs, train_mlp
from src.models import save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--encoder", default="dino_register")
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    data = load_dataset(args.dataset, args.encoder, args.config, targets=args.targets)
    model, payload = train_mlp(
        data.features,
        data.targets,
        args.targets,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        val_split=args.val_split,
        device=args.device,
    )
    payload.update(dataset=args.dataset, encoder=args.encoder, sample_keys=data.metadata.key.tolist())
    save_checkpoint(payload, args.output)
    ids = payload["validation_indices"]
    with torch.no_grad():
        prediction = forward_outputs(model, data.x[ids].to(args.device), args.targets).cpu().numpy()
    prediction = prediction * payload["target_std"] + payload["target_mean"]
    metrics = {k: v.tolist() for k, v in regression_metrics(data.targets[ids], prediction).items()}
    args.output.with_suffix(".json").write_text(
        json.dumps(
            dict(dataset=args.dataset, targets=args.targets, seed=args.seed, validation=metrics), indent=2
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
