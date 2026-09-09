from pathlib import Path
from .timing import BlockTimer
from .prepared import PreparedImages
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
        if isinstance(image, PreparedImages):
            transformed = image.sam
            if transformed.shape[-2:] != (self.sam.image_encoder.img_size,)*2:
                raise ValueError("Prepared SAM image size does not match this encoder")
        else:
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


class FiLMSpatialDecoder(nn.Module):
    """Trainable spatial head conditioned by the global bottleneck readout.

    It predicts logits directly from dense frozen features.  This avoids asking
    SAM's prompt-conditioned instance mask decoder to act as a semantic
    segmentation decoder without prompts.
    """
    def __init__(self, channels, readout_dim, classes, width=192, dropout=.1, use_image_refiner=True):
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        self.use_image_refiner = use_image_refiner
        self.stem = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1, bias=False), nn.GroupNorm(groups, width), nn.GELU())
        self.film = nn.Linear(readout_dim, 2*width)
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False), nn.GroupNorm(groups, width), nn.GELU(),
            nn.Dropout2d(dropout), nn.Conv2d(width, width, 3, padding=1, bias=False), nn.GroupNorm(groups, width), nn.GELU(),
        )
        if use_image_refiner:
            refine_width = max(32, width//2)
            refine_groups = 8 if refine_width % 8 == 0 else 1
            self.image = nn.Sequential(nn.Conv2d(3, refine_width, 3, padding=1, bias=False), nn.GroupNorm(refine_groups, refine_width), nn.GELU())
            self.refine = nn.Sequential(
                nn.Conv2d(width+refine_width, refine_width, 3, padding=1, bias=False), nn.GroupNorm(refine_groups, refine_width), nn.GELU(),
                nn.Conv2d(refine_width, classes, 1),
            )
        else:
            self.classifier = nn.Conv2d(width, classes, 1)

    def forward(self, features, readout, image, output_size):
        x = self.stem(features)
        scale, shift = self.film(readout).chunk(2, dim=1)
        x = x * (1 + scale.tanh()[..., None, None]) + shift[..., None, None]
        x = x + self.body(x)
        x = F.interpolate(x, output_size, mode="bilinear", align_corners=False)
        if self.use_image_refiner:
            x = self.refine(torch.cat((x, self.image(image)), dim=1))
        else:
            x = self.classifier(x)
        return x


class SpatialLoRAAdapter(nn.Module):
    """Low-rank residual adapter on frozen SAM spatial features."""
    def __init__(self, channels, rank=16):
        super().__init__()
        self.down = nn.Conv2d(channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, features):
        return features + self.scale * self.up(F.gelu(self.down(features)))


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
        self.decoder_mode = cfg.get("decoder_mode", "sam_prompt")
        if self.decoder_mode not in ("sam_prompt", "spatial_fpn"):
            raise ValueError("decoder_mode must be sam_prompt or spatial_fpn")
        # The prompt projection exists only for the frozen-SAM ablation.  The
        # spatial decoder owns every trainable segmentation parameter in the
        # accuracy configuration, so unused prompt weights cannot silently
        # consume optimizer capacity.
        self.decoder_head = None
        self.spatial_decoder_head = None
        if self.decoder_mode == "sam_prompt":
            self.decoder_head = nn.Linear(self.bottleneck.readout_dim, self.classes*self.backbone.channels)
        if self.decoder_mode == "sam_prompt" and cfg.get("decoder_spatial_adapter", False):
            self.spatial_decoder_head = nn.Sequential(
                nn.Conv2d(self.backbone.channels, self.backbone.channels, 1),
                nn.GELU(),
                nn.Conv2d(self.backbone.channels, self.classes*self.backbone.channels, 1),
            )
        self.spatial_decoder = None
        self.feature_adapter = None
        if int(cfg.get("adapter_rank", 0)) > 0:
            self.feature_adapter = SpatialLoRAAdapter(self.backbone.channels, int(cfg["adapter_rank"]))
        if self.decoder_mode == "spatial_fpn":
            self.spatial_decoder = FiLMSpatialDecoder(
                self.backbone.channels, self.bottleneck.readout_dim, self.classes,
                width=int(cfg.get("decoder_width", 192)), dropout=float(cfg.get("decoder_dropout", .1)),
                use_image_refiner=bool(cfg.get("decoder_image_refiner", True)),
            )

    def representation(self, images):
        z = self.backbone.encode(images)
        if self.feature_adapter is not None:
            z = self.feature_adapter(z)
        return normalize(self.projection(z))

    def forward(self, images):
        timer = getattr(self, "timer", None) or BlockTimer()
        with timer.block('model: frozen image encoder'):
            z = self.backbone.encode(images)
        if isinstance(images, PreparedImages):
            images = images.image
        if self.feature_adapter is not None:
            with timer.block('model: spatial feature adapter'):
                z = self.feature_adapter(z)
        with timer.block('model: projection and normalization'):
            h = normalize(self.projection(z))
        with timer.block('model: bottleneck'):
            r = self.bottleneck(h)
        if self.decoder_mode == "spatial_fpn":
            with timer.block('model: spatial decoder'):
                return self.spatial_decoder(z, r, images, images.shape[-2:])
        with timer.block('model: prompt projection'):
            prompts = self.decoder_head(r).reshape(len(images), self.classes, self.backbone.channels)
        if self.spatial_decoder_head is not None:
            with timer.block('model: spatial prompt adapter'):
                spatial = self.spatial_decoder_head(z).reshape(len(images), self.classes, self.backbone.channels, *z.shape[-2:])
            prompts = spatial + prompts[..., None, None]
        with timer.block('model: SAM prompt decoder'):
            return self.backbone.decode(z, prompts, images.shape[-2:])
