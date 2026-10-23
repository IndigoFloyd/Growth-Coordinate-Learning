"""Frozen RGB encoders with the preprocessing used by the paper."""

from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from .models import set_global_seed

# Pin the published model weights as well as the Python dependencies.
MODEL_SPECS = {
    "dino_register": ("facebook/dinov2-with-registers-base", "a1d738ccfa7ae170945f210395d99dde8adb1805"),
    "clip_vitb32": ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"),
    "mae": ("facebook/vit-mae-base", "25b184bea5538bf5c4c852c79d221195fdd2778d"),
    "vae": ("stabilityai/sd-vae-ft-mse", "31f26fdeee1355a5c34592e401dd41e45d25a493"),
}
ENCODERS = (*MODEL_SPECS, "resnet50")


def image_transform(size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return transforms.Compose([
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class FrozenEncoder:
    def __init__(self, name, device="cuda", model_path=None, checkpoint=None, *, cudnn_tf32=False):
        set_global_seed(42)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        self.name = name
        self.device = torch.device(device)
        if checkpoint is not None and name not in ("dino_register", "resnet50"):
            raise ValueError("--checkpoint is supported for DINOv2 and ResNet50 only.")
        if name == "resnet50" and model_path is not None:
            raise ValueError("Use --checkpoint for a local ResNet50 state dictionary.")
        if name in MODEL_SPECS:
            model_id, revision = MODEL_SPECS[name]
            source = str(model_path) if model_path is not None else model_id
            options = {} if model_path is not None else {"revision": revision}
        self.transform = image_transform()
        if name == "dino_register":
            from transformers import AutoImageProcessor, Dinov2WithRegistersModel

            self.model = Dinov2WithRegistersModel.from_pretrained(
                source, **options, use_safetensors=True, attn_implementation="eager"
            )
            # The original extraction used a non-affine final LayerNorm.
            self.model.layernorm.elementwise_affine = False
            self.model.layernorm.weight = None
            self.model.layernorm.bias = None
            processor = AutoImageProcessor.from_pretrained(source, **options, use_fast=False)
            self.transform = image_transform(mean=processor.image_mean, std=processor.image_std)
        elif name == "clip_vitb32":
            from transformers import CLIPImageProcessor, CLIPVisionModel

            self.processor = CLIPImageProcessor.from_pretrained(source, **options)
            self.model = CLIPVisionModel.from_pretrained(
                source, **options, attn_implementation="eager"
            )
        elif name == "mae":
            from transformers import AutoImageProcessor, ViTMAEModel

            self.model = ViTMAEModel.from_pretrained(source, **options, use_safetensors=True)
            processor = AutoImageProcessor.from_pretrained(source, **options, use_fast=False)
            size = processor.size["height"]
            self.transform = image_transform(size, processor.image_mean, processor.image_std)
        elif name == "vae":
            from diffusers import AutoencoderKL

            self.model = AutoencoderKL.from_pretrained(source, **options, torch_dtype=torch.float32)
            self.transform = image_transform(256, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif name == "resnet50":
            weights = None if checkpoint is not None else models.ResNet50_Weights.IMAGENET1K_V2
            self.model = models.resnet50(weights=weights)
            # Keep the paper's cached-weight path: bicubic resize to 224,
            # rather than switching preprocessing when weights are downloaded.
        else:
            raise ValueError(f"Unknown encoder: {name}")
        if checkpoint is not None:
            state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
            self.model.load_state_dict(state.get("state_dict", state))
        if name == "resnet50":
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.requires_grad_(False).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.name == "clip_vitb32":
                tensor = self.processor(images=image, return_tensors="pt").pixel_values
            else:
                tensor = self.transform(image).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=torch.float32)
        if self.name == "vae":
            features = self.model.encode(tensor).latent_dist.mean.flatten(start_dim=1)
        elif self.name == "resnet50":
            features = self.model(tensor)
        else:
            features = self.model(pixel_values=tensor).last_hidden_state
            if self.name == "dino_register":
                features = features[:, 1 + self.model.config.num_register_tokens:]
            elif self.name == "mae":
                features = features[:, 1:]
        return features.detach().cpu()
