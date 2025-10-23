import os
import torch
import torch.nn as nn

# NEW: Hugging Face DINOv3
from transformers import AutoImageProcessor, AutoModel

from networks.backbones.mylora import Linear as LoraLinear
from networks.backbones.mylora import DVLinear as DVLinear
from .layers import HeadDepth
from .layers import mark_only_part_as_trainable,_make_scratch, _make_fusion_block


# ---------------------------
# Adapter to make HF DINOv3 look like your old ViT backbone
# ---------------------------
class Dinov3Backbone(nn.Module):
    """
    Wraps HF DINOv3 to provide:
      - get_intermediate_layers(n, return_class_token=True)
      - .blocks iterable for LoRA replacement
      - attributes: embed_dim, patch_size
    """
    def __init__(self, model_id: str, include_cls_token: bool = True, image_size=(224, 288)):
        super().__init__()
        self.include_cls_token = include_cls_token

        # Processor is optional if you already normalize upstream; kept for reference.
        # For training with your existing normalization, you can skip using it.
        self.processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,           # several DINOv3 repos use custom code
            output_hidden_states=True         # <-- we need hidden states
        )

        # Expose a blocks-like list to ease LoRA injection
        # HF typically has encoder blocks under .encoder.layer or .blocks
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            self.blocks = self.model.encoder.layer
        elif hasattr(self.model, "blocks"):
            self.blocks = self.model.blocks
        else:
            raise RuntimeError("Unexpected DINOv3 model structure; cannot find transformer blocks.")

        # Dimensions & patch size
        cfg = getattr(self.model, "config", None)
        self.embed_dim = getattr(cfg, "hidden_size", None) or getattr(cfg, "embed_dim", None)
        self.patch_size = getattr(cfg, "patch_size", 16)   # DINOv3 commonly uses 16

        # track image size if you want to sanity-check divisibility
        self.image_size = image_size

    @torch.no_grad()
    def _take_intermediate(self, hidden_states, num_layers: int, return_class_token: bool):
        """
        hidden_states: list of tensors [embeddings, layer1, layer2, ..., layerN]
          each tensor is (B, seq_len, C) with CLS at index 0 if present.
        Return last `num_layers` states as [(tokens_wo_cls, cls_token), ...]
        """
        # Take the last `num_layers` states (excluding the embeddings at index 0)
        hs = hidden_states[1:][-num_layers:]

        outs = []
        for h in hs:
            # h: (B, seq, C), seq = 1 + N_patches if CLS present
            if return_class_token:
                cls_token = h[:, 0, :]                    # (B, C)
                tokens = h[:, 1:, :]                      # (B, N, C)
                outs.append((tokens, cls_token))
            else:
                tokens = h[:, 1:, :] if self.include_cls_token else h
                outs.append((tokens, None))
        return outs

    def get_intermediate_layers(self, x, n: int, return_class_token: bool = True):
        """
        x: (B, 3, H, W) tensor already normalized/resized upstream.
        returns: list of tuples [(tokens, cls), ...] with length n
        """
        outputs = self.model(pixel_values=x)
        hidden_states = outputs.hidden_states  # list of (B, seq, C)
        return self._take_intermediate(hidden_states, n, return_class_token)


# ---------------------------
# Your DPT head (unchanged)
# ---------------------------
class DPTHead(nn.Module):
    def __init__(self, in_channels, features=128, use_bn=False, out_channels=[96, 192, 384, 768], use_clstoken=False):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.conv_depth_1 = HeadDepth(features)
        self.conv_depth_2 = HeadDepth(features)
        self.conv_depth_3 = HeadDepth(features)
        self.conv_depth_4 = HeadDepth(features)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        outputs = {}
        outputs[("disp", 3)] = self.sigmoid(self.conv_depth_4(path_4))
        outputs[("disp", 2)] = self.sigmoid(self.conv_depth_3(path_3))
        outputs[("disp", 1)] = self.sigmoid(self.conv_depth_2(path_2))
        outputs[("disp", 0)] = self.sigmoid(self.conv_depth_1(path_1))

        return outputs


# ---------------------------
# Main model: now using DINOv3
# ---------------------------
class endodac(nn.Module):
    """
    Applies low-rank adaptation to a DINOv3 ViT image encoder.

    Args:
        backbone_size: "small" or "base" (maps to DINOv3 ViT-S/16 or ViT-B/16)
        r: rank of LoRA
        image_shape: input HxW; must be divisible by the DINOv3 patch size (typically 16)
        lora_type: "lora", "dvlora", or "none"
        residual_block_indexes: kept for compatibility (unused by DINOv3)
        include_cls_token: keep CLS token in the stream
        use_cls_token: use CLS readout in the DPT head
        use_bn: batch norm inside fusion blocks
        pretrained_path: kept for backward-compat (Depth Anything weights). Not used with DINOv3.
    """

    def __init__(self, 
                 backbone_size = "base", 
                 r=4, 
                 image_shape=(224, 288), 
                 lora_type="lora",
                 pretrained_path=None,            # not used for DINOv3
                 residual_block_indexes=[],
                 include_cls_token=True,
                 use_cls_token=False,
                 use_bn=False,
                 dinov3_ids=None):
        super(endodac, self).__init__()

        assert r > 0
        self.r = r
        self.backbone_size = backbone_size

        # Map sizes to HF model IDs (override via dinov3_ids if you like)
        default_ids = {
            "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "base":  "facebook/dinov3-vitb16-pretrain-lvd1689m",
        }
        if dinov3_ids is not None:
            default_ids.update(dinov3_ids)

        # Create the DINOv3 backbone
        encoder = Dinov3Backbone(
            model_id=default_ids[self.backbone_size],
            include_cls_token=include_cls_token,
            image_size=image_shape
        )

        # Expose dimensions/out channels (match your old layout)
        self.embedding_dims = {
            "small": encoder.embed_dim,   # typically 384
            "base":  encoder.embed_dim,   # typically 768
        }
        self.depth_head_features = {
            "small": 64,
            "base": 128,
        }
        self.depth_head_out_channels = {
            "small": [48, 96, 192, self.embedding_dims["small"]],
            "base":  [96, 192, 384, self.embedding_dims["base"]],
        }

        self.embedding_dim = self.embedding_dims[self.backbone_size]
        self.depth_head_feature = self.depth_head_features[self.backbone_size]
        self.depth_head_out_channel = self.depth_head_out_channels[self.backbone_size]

        self.encoder = encoder
        self.image_shape = image_shape

        # ---- Apply LoRA / DVLoRA on MLP linears inside each transformer block
        if lora_type != "none":
            for blk in self.encoder.blocks:
                # replace all Linear layers in the MLP submodules
                for name, module in list(blk.named_modules()):
                    # heuristically target MLP dense layers (avoid qkv proj unless desired)
                    if isinstance(module, nn.Linear) and ("mlp" in name or "intermediate" in name):
                        in_features = module.in_features
                        out_features = module.out_features
                        bias = module.bias is not None
                        if lora_type == "dvlora":
                            new_lin = DVLinear(in_features, out_features, r=self.r, lora_alpha=self.r, bias=bias)
                        else:
                            new_lin = LoraLinear(in_features, out_features, r=self.r, bias=bias)

                        # Set weights from the original module for a warm start
                        new_lin.weight.data.copy_(module.weight.data)
                        if bias:
                            new_lin.bias.data.copy_(module.bias.data)

                        # write back into the block
                        parent = blk
                        # navigate to the parent of 'name'
                        parts = name.split(".")
                        for p in parts[:-1]:
                            parent = getattr(parent, p)
                        setattr(parent, parts[-1], new_lin)

        self.depth_head = DPTHead(
            self.embedding_dim,
            self.depth_head_feature,
            use_bn,
            out_channels=self.depth_head_out_channel,
            use_clstoken=use_cls_token
        )

        # Depth Anything .pth loading is NOT applicable to DINOv3; ignore if provided.
        if pretrained_path is not None:
            print("Note: 'pretrained_path' is ignored for DINOv3. Using HF pre-trained weights instead.")

        mark_only_part_as_trainable(self.encoder)
        mark_only_part_as_trainable(self.depth_head)

    def forward(self, pixel_values):
        # Ensure size is divisible by the DINOv3 patch size
        patch = self.encoder.patch_size
        H = (self.image_shape[0] // patch) * patch
        W = (self.image_shape[1] // patch) * patch
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=(H, W), mode="bilinear", align_corners=True)

        h, w = pixel_values.shape[-2:]
        features = self.encoder.get_intermediate_layers(pixel_values, 4, return_class_token=True)

        # Use the true patch size (v3 is usually 16)
        patch_h, patch_w = h // patch, w // patch

        disp = self.depth_head(features, patch_h, patch_w)
        return disp
