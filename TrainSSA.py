"""Train SSA for a configured target dataset using any compatible source MLP."""

import argparse
from pathlib import Path

from src.dataloader import CONFIG, load_dataset
from src.models import save_checkpoint
from src.ssa import train_ssa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--encoder", default="dino_register")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["ln", "lora", "ln_lora"], default="ln_lora")
    parser.add_argument(
        "--reference", choices=["source", "target_mean", "target_variance", "target_both"], default="source"
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--before", help="Optional timestamp cutoff for adaptation samples.")
    parser.add_argument("--snapshot-steps", type=int, nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    source = load_dataset(args.source, args.encoder, args.config, targets=[])
    target = load_dataset(args.target, args.encoder, args.config, targets=[], before=args.before)

    def snapshot(model, step, statistics, loss):
        if step in args.snapshot_steps:
            save_checkpoint(
                dict(
                    state_dict=model.state_dict(),
                    seed=args.seed,
                    reference=args.reference,
                    step=step,
                    batch_size=args.batch_size,
                    top_k=args.top_k,
                    target_names=getattr(model, "output_metrics", None),
                ),
                args.output.parent / (args.output.stem + "-steps") / f"step{step}.pth",
            )

    _, payload = train_ssa(
        source.features,
        target.features,
        args.source_model,
        epochs=args.epochs,
        lr=args.lr,
        rank=args.rank,
        top_k=args.top_k,
        seed=args.seed,
        batch_size=args.batch_size,
        mode=args.mode,
        shuffle=args.shuffle,
        reference=args.reference,
        device=args.device,
        step_callback=snapshot if args.snapshot_steps else None,
    )
    payload.update(
        source_dataset=args.source,
        target_dataset=args.target,
        encoder=args.encoder,
        fit_before=args.before,
        target_keys=target.metadata.key.tolist(),
    )
    save_checkpoint(payload, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
