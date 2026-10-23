"""Extract frozen image features for TrainMLP, TrainSSA, and figure/table scripts."""

import argparse
import hashlib
from pathlib import Path

import torch
from tqdm import tqdm

from src.dataloader import CONFIG, dataset_config, extraction_inputs, pooled_vector, raw_vector
from src.encoders import ENCODERS, FrozenEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--encoder", choices=ENCODERS, default="dino_register")
    parser.add_argument("--output-dir", type=Path, help="Override this dataset's feature directory.")
    parser.add_argument("--model-path", type=Path, help="Local Hugging Face model directory.")
    parser.add_argument("--checkpoint", type=Path, help="Optional DINOv2 or ResNet50 state dictionary.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing feature files.")
    args = parser.parse_args()
    inputs = extraction_inputs(args.dataset, args.encoder, args.config, args.output_dir)
    if not inputs:
        raise ValueError(f"No input images found for {args.dataset}.")
    existing = next((output for _, output, _ in inputs if output.exists()), None)
    if existing is not None and not args.overwrite:
        raise FileExistsError(f"{existing} already exists; use a new --output-dir or --overwrite.")
    cfg = dataset_config(args.dataset, args.config)
    encoder = FrozenEncoder(
        args.encoder, args.device, args.model_path, args.checkpoint,
        cudnn_tf32=cfg.get("cudnn_tf32", False),
    )
    compact = cfg.get("format") == "gromo"
    for image, output, key in tqdm(inputs, desc=f"{args.encoder}:{args.dataset}", unit="image"):
        # Stable per-image MAE masks; independent of iteration order or other images.
        if args.encoder == "mae":
            torch.manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
        features = encoder(image)
        if compact:
            features = {
                "raw": torch.from_numpy(raw_vector(features).copy()),
                "ssa": torch.from_numpy(pooled_vector(features).copy()),
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, output)
    print(f"Saved {len(inputs)} image features for {args.dataset}.")


if __name__ == "__main__":
    main()
