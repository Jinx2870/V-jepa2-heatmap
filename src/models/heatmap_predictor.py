# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils.modules import ACBlock as Block
from src.models.utils.modules import build_action_block_causal_attention_mask
from src.utils.tensors import trunc_normal_


class VisionTransformerPredictorHeatmap(nn.Module):
    """Gaze Heatmap Conditioned Vision Transformer Predictor"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=True,
        use_silu=False,
        wide_silu=True,
        is_frame_causal=True,
        use_activation_checkpointing=False,
        use_rope=True,
        heatmap_embed_dim=None,  # If None, will be computed from heatmap size
        heatmap_patch_size=16,  # Patch size for heatmap processing
        **kwargs
    ):
        super().__init__()
        self.is_frame_causal = is_frame_causal

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        
        # Heatmap processing
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.heatmap_patch_size = heatmap_patch_size
        
        # Compute heatmap embedding dimension
        # Heatmap is typically [T, H, W] where H=W=224 (or crop_size)
        # We'll process it similar to image patches
        heatmap_grid_h = self.img_height // self.heatmap_patch_size
        heatmap_grid_w = self.img_width // self.heatmap_patch_size
        heatmap_patches_per_frame = heatmap_grid_h * heatmap_grid_w
        
        # Use a CNN to extract features from heatmap, then project to predictor_embed_dim
        # Or use patch embedding similar to vision transformer
        # For simplicity, we'll use a linear projection after flattening patches
        # You can also use a small CNN here
        self.heatmap_patch_embed = nn.Conv2d(
            1, predictor_embed_dim, 
            kernel_size=heatmap_patch_size, 
            stride=heatmap_patch_size
        )
        # Alternative: use linear projection
        # self.heatmap_encoder = nn.Linear(heatmap_patches_per_frame, predictor_embed_dim, bias=True)
        
        # Determine positional embedding
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Position embedding
        self.uniform_power = uniform_power

        # Attention Blocks
        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ------ initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        attn_mask = None
        if self.is_frame_causal:
            grid_depth = self.num_frames // self.tubelet_size
            grid_height = self.img_height // self.patch_size
            grid_width = self.img_width // self.patch_size
            # We add 1 token per frame for heatmap
            attn_mask = build_action_block_causal_attention_mask(
                grid_depth, grid_height, grid_width, add_tokens=1
            )
        self.attn_mask = attn_mask

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def process_heatmap(self, heatmap):
        """
        Process heatmap to extract features
        :param heatmap: [B, T, H, W] or [B, T, 1, H, W] heatmap tensor
        :return: [B, T, D] heatmap features
        """
        B, T = heatmap.shape[:2]
        
        # Ensure heatmap is [B*T, 1, H, W]
        if len(heatmap.shape) == 4:
            # [B, T, H, W] -> [B*T, 1, H, W]
            heatmap = heatmap.unsqueeze(2)  # [B, T, 1, H, W]
        heatmap = heatmap.view(B * T, 1, self.img_height, self.img_width)
        
        # Extract patch features using CNN
        # Output: [B*T, D, H_patches, W_patches]
        heatmap_features = self.heatmap_patch_embed(heatmap)
        
        # Global average pooling to get one token per frame
        # [B*T, D, H_patches, W_patches] -> [B*T, D]
        heatmap_features = F.adaptive_avg_pool2d(heatmap_features, (1, 1)).squeeze(-1).squeeze(-1)
        
        # Reshape back to [B, T, D]
        heatmap_features = heatmap_features.view(B, T, -1)
        
        return heatmap_features

    def forward(self, x, heatmap=None):
        """
        :param x: context tokens [B, N, D]
        :param heatmap: [B, T, H, W] or [B, T, 1, H, W] heatmap tensor
        """
        # Map tokens to predictor dimensions
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        T = N_ctxt // (self.grid_height * self.grid_width)

        # Process heatmap if provided
        if heatmap is not None:
            heatmap_tokens = self.process_heatmap(heatmap)  # [B, T, D]
            heatmap_tokens = heatmap_tokens.unsqueeze(2)  # [B, T, 1, D]
            
            # Reshape x to [B, T, H*W, D]
            x = x.view(B, T, self.grid_height * self.grid_width, D)
            
            # Concatenate heatmap token with frame tokens
            # [B, T, 1+H*W, D]
            x = torch.cat([heatmap_tokens, x], dim=2).flatten(1, 2)  # [B, T*(1+H*W), D]
            
            cond_tokens = 1
        else:
            # No heatmap, just use frame tokens
            # Reshape x to [B, T, H*W, D] for consistency
            x = x.view(B, T, self.grid_height * self.grid_width, D).flatten(1, 2)
            cond_tokens = 0

        attn_mask = None
        if self.attn_mask is not None:
            attn_mask = self.attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)

        # Fwd prop
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                )

        # Split out heatmap and frame tokens
        if cond_tokens > 0:
            x = x.view(B, T, cond_tokens + self.grid_height * self.grid_width, D)  # [B, T, 1+H*W, D]
            x = x[:, :, cond_tokens:, :].flatten(1, 2)  # [B, T*H*W, D]

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        return x


def vit_heatmap_predictor(**kwargs):
    model = VisionTransformerPredictorHeatmap(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

