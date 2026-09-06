from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .bottlenecks import build_bottleneck, normalize


class FrozenSAM(nn.Module):
    """Frozen weights, but decoder operations keep gradients to learned prompts."""
    channels = 256

    def __init__(self, checkpoint, variant="vit_b"):
        super().__init__()
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"SAM checkpoint missing: {checkpoint}")
        from segment_anything import sam_model_registry
        from segment_anything.utils.transforms import ResizeLongestSide
        self.sam = sam_model_registry[variant](checkpoint=checkpoint)
        self.sam.requires_grad_(False).eval()
        self.resize = ResizeLongestSide(self.sam.image_encoder.img_size)
        assert not any(p.requires_grad for p in self.sam.parameters())

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, image):
        transformed = self.resize.apply_image_torch(image * 255.)
        return self.sam.image_encoder(self.sam.preprocess(transformed))

    def decode(self, features, prompts, output_size):
        # Official SAM decoder expands one image for N prompts. Process images
        # separately to avoid its repeat_interleave creating B*B images.
        outputs = []
        size = self.resize.get_preprocess_shape(*output_size, self.sam.image_encoder.img_size)
        for embedding, prompt in zip(features, prompts):
            channels = prompt.shape[0]
            sparse = embedding.new_empty((channels, 0, self.channels))
            if prompt.ndim == 2:
                dense = prompt[..., None, None].expand(-1, -1, *embedding.shape[-2:])
            elif prompt.ndim == 4 and prompt.shape[1:] == embedding.shape:
                dense = prompt
            else:
                raise ValueError(f"Dense prompt must be [classes, channels] or [classes, channels, height, width], got {tuple(prompt.shape)}")
            dense = dense + self.sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
            logits, _ = self.sam.mask_decoder(
                image_embeddings=embedding[None], image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
            logits = self.sam.postprocess_masks(logits, size, output_size)
            outputs.append(logits[:, 0])
        return torch.stack(outputs)


class TinyBackbone(nn.Module):
    """Random frozen surrogate for fast plumbing tests; never a SAM accuracy result."""
    channels = 32

    def __init__(self, **_):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, stride=2), nn.GELU(),
                                     nn.Conv2d(16, 32, 3, padding=1, stride=2), nn.GELU())
        self.decoder = nn.Conv2d(32, 1, 1)
        self.requires_grad_(False)

    @torch.no_grad()
    def encode(self, image):
        return self.encoder(image)

    def decode(self, features, prompts, output_size):
        conditioned = features[:, None] + (prompts[..., None, None] if prompts.ndim == 3 else prompts)
        b, k, c, h, w = conditioned.shape
        logits = self.decoder(conditioned.reshape(b*k, c, h, w))
        return F.interpolate(logits, output_size, mode="bilinear", align_corners=False).reshape(b, k, *output_size)


class TQSI(nn.Module):
    """forward: float RGB [B,3,H,W] in [0,1] -> raw logits [B,K,H,W]."""
    def __init__(self, cfg, n_tasks):
        super().__init__()
        if cfg["backbone"] not in ("sam", "tiny"):
            raise ValueError("backbone must be explicitly sam or tiny")
        self.classes = cfg.get("num_classes", 1)
        self.backbone = FrozenSAM(cfg["sam_checkpoint"], cfg.get("sam_variant", "vit_b")) if cfg["backbone"] == "sam" else TinyBackbone()
        self.bottleneck = build_bottleneck(cfg, n_tasks)
        self.projection = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(self.backbone.channels, self.bottleneck.feature_dim), nn.LayerNorm(self.bottleneck.feature_dim))
        self.decoder_head = nn.Linear(self.bottleneck.readout_dim, self.classes*self.backbone.channels)
        self.spatial_decoder_head = None
        if cfg.get("decoder_spatial_adapter", False):
            self.spatial_decoder_head = nn.Sequential(
                nn.Conv2d(self.backbone.channels, self.backbone.channels, 1),
                nn.GELU(),
                nn.Conv2d(self.backbone.channels, self.classes*self.backbone.channels, 1),
            )

    def representation(self, images):
        return normalize(self.projection(self.backbone.encode(images)))

    def forward(self, images):
        z = self.backbone.encode(images)
        h = normalize(self.projection(z))
        r = self.bottleneck(h)
        prompts = self.decoder_head(r).reshape(len(images), self.classes, self.backbone.channels)
        if self.spatial_decoder_head is not None:
            spatial = self.spatial_decoder_head(z).reshape(len(images), self.classes, self.backbone.channels, *z.shape[-2:])
            prompts = spatial + prompts[..., None, None]
        return self.backbone.decode(z, prompts, images.shape[-2:])
