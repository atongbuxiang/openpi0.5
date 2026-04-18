# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Short-term video memory encoder for pi0.5."""

from collections.abc import Sequence

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.siglip as siglip
import openpi.training.sharding as sharding


def temporal_posemb_sincos(length: int, width: int, dtype=jnp.float32):
    """Sine-cosine temporal embedding with the current frame anchored at zero."""
    if width % 2 != 0:
        raise ValueError(f"Width must be divisible by 2 for temporal sincos, got {width}")
    pos = jnp.arange(length, dtype=jnp.float32) - (length - 1)
    omega = jnp.arange(width // 2, dtype=jnp.float32)
    omega = 1.0 / (10_000.0 ** (omega / jnp.maximum(width // 2 - 1, 1)))
    angles = jnp.einsum("t,d->td", pos, omega)
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    emb = emb.at[-1].set(0.0)
    return emb.astype(dtype)


class TemporalCausalAttention(nn.Module):
    """Causal temporal attention applied independently per spatial patch."""

    width: int
    num_heads: int
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, frame_mask, deterministic=True):  # noqa: FBT002
        b, t, n, d = x.shape
        x = sharding.activation_sharding_constraint(x)
        x = x + temporal_posemb_sincos(t, d, dtype=jnp.float32)[None, :, None, :]
        y = nn.LayerNorm(dtype=self.dtype_mm, name="ln")(x)
        y = einops.rearrange(y, "b t n d -> (b n) t d")
        temporal_mask = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))
        valid = einops.repeat(frame_mask.astype(jnp.bool_), "b t -> (b n) t", n=n)
        valid_q = valid[:, None, :, None]
        valid_k = valid[:, None, None, :]
        attn_mask = jnp.logical_and(temporal_mask[None, None, :, :], valid_q)
        attn_mask = jnp.logical_and(attn_mask, valid_k)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
            name="attn",
        )(y, y, mask=attn_mask)
        y = einops.rearrange(y, "(b n) t d -> b t n d", b=b, n=n)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        return x + y


class SpaceTimeSeparableEncoderBlock(nn.Module):
    """Interleaves causal temporal attention with spatial SigLIP attention."""

    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, frame_mask, deterministic=True):  # noqa: FBT002
        b, t, n, d = x.shape
        x = TemporalCausalAttention(
            width=d,
            num_heads=self.num_heads,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
            name="temporal",
        )(x, frame_mask, deterministic=deterministic)

        y = einops.rearrange(x, "b t n d -> (b t) n d")
        y = sharding.activation_sharding_constraint(y)
        y_ln = nn.LayerNorm(dtype=self.dtype_mm, name="spatial_ln")(y)
        y_sa = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
            name="spatial_attn",
        )(y_ln, y_ln)
        y_sa = sharding.activation_sharding_constraint(y_sa)
        y = y + nn.Dropout(rate=self.dropout)(y_sa, deterministic)

        y_mlp = nn.LayerNorm(dtype=self.dtype_mm, name="mlp_ln")(y)
        y_mlp = siglip.MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
            name="mlp",
        )(y_mlp, deterministic)
        y = y + nn.Dropout(rate=self.dropout)(y_mlp, deterministic)
        y = sharding.activation_sharding_constraint(y)
        return einops.rearrange(y, "(b t) n d -> b t n d", b=b, t=t), {}


class VideoEncoder(nn.Module):
    depth: int
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, frame_mask, deterministic=True):  # noqa: FBT002
        out = {}
        for lyr in range(self.depth):
            x, out[f"block{lyr:02d}"] = SpaceTimeSeparableEncoderBlock(
                name=f"encoderblock_{lyr}",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, frame_mask, deterministic)
        x = einops.rearrange(x, "b t n d -> (b t) n d")
        x = nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x)
        return einops.rearrange(x, "(b t) n d -> b t n d", b=frame_mask.shape[0], t=frame_mask.shape[1]), out


class _VideoModule(nn.Module):
    """Video extension of the SigLIP vision encoder for short-term memory."""

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None
    num_heads: int = 12
    posemb: str = "learn"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "none"
    head_zeroinit: bool = True
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    history_pool_tokens: int = 8
    keep_full_current_tokens: bool = True

    @nn.compact
    def __call__(self, video, frame_mask, *, train=False):
        if video.ndim != 5:
            raise ValueError(f"Video input must be [B, T, H, W, C], got {video.shape}")

        out = {}
        b, t, _, _, _ = video.shape
        flat_video = einops.rearrange(video, "b t h w c -> (b t) h w c")
        flat_video = jnp.asarray(flat_video, jnp.float32)

        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(flat_video)

        _, h, w, c = x.shape
        x = jnp.reshape(x, [b, t, h * w, c])
        x = x + siglip.get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)[:, None, :, :]
        x = nn.Dropout(rate=self.dropout)(x, not train)
        x = x.astype(self.dtype_mm)

        x, out["encoder"] = VideoEncoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, frame_mask, deterministic=not train)

        current_tokens = x[:, -1]
        if t == 1:
            encoded = current_tokens
        else:
            history = x[:, :-1]
            history_mask = frame_mask[:, :-1]
            pooled_history = self._pool_history(history, history_mask)
            encoded = jnp.concatenate([pooled_history, current_tokens], axis=1) if self.keep_full_current_tokens else pooled_history

        x_2d = jnp.reshape(x[:, -1], [b, h, w, -1])
        out["encoded"] = encoded
        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = encoded

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            encoded = out["logits"] = head(encoded)

        return encoded, out

    def _pool_history(self, history, history_mask):
        b, t, n, d = history.shape
        valid = history_mask.astype(history.dtype)[..., None, None]
        history = history * valid
        flat = einops.rearrange(history, "b t n d -> b (t n) d")
        flat_valid = einops.repeat(history_mask.astype(history.dtype), "b t -> b (t n) 1", n=n)

        if self.history_pool_tokens <= 0:
            denom = jnp.maximum(flat_valid.sum(axis=1, keepdims=True), 1.0)
            return (flat * flat_valid).sum(axis=1, keepdims=True) / denom

        total_tokens = flat.shape[1]
        group_size = max(total_tokens // self.history_pool_tokens, 1)
        pooled = []
        for i in range(self.history_pool_tokens):
            start = i * group_size
            end = total_tokens if i == self.history_pool_tokens - 1 else min((i + 1) * group_size, total_tokens)
            if start >= total_tokens:
                pooled.append(jnp.zeros((b, 1, d), dtype=flat.dtype))
                continue
            cur = flat[:, start:end]
            cur_valid = flat_valid[:, start:end]
            denom = jnp.maximum(cur_valid.sum(axis=1, keepdims=True), 1.0)
            pooled.append((cur * cur_valid).sum(axis=1, keepdims=True) / denom)
        return jnp.concatenate(pooled, axis=1)


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    return _VideoModule(num_classes, **{**siglip.decode_variant(variant), **kw})
