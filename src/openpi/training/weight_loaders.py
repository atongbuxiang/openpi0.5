import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    _bootstrap_video_memory_params(flat_loaded, flat_ref)

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _bootstrap_video_memory_params(flat_loaded: dict[str, np.ndarray], flat_ref: dict[str, np.ndarray]) -> None:
    """Warm-start memory_img from img weights when loading old checkpoints.

    Old pi0/pi0.5 checkpoints do not contain `PaliGemma/memory_img/*`. For the
    new short-memory model we can still initialize a large subset of the video
    encoder from the 2D SigLIP encoder:
    - stem / pos_embedding / head copy directly
    - spatial attention + MLP blocks copy from the corresponding SigLIP block
    - temporal-only layers remain randomly initialized from the reference params
    """
    memory_prefix = "PaliGemma/memory_img/"
    image_prefix = "PaliGemma/img/"
    if any(k.startswith(memory_prefix) for k in flat_loaded):
        return
    if not any(k.startswith(image_prefix) for k in flat_loaded):
        return

    for ref_key, ref_value in flat_ref.items():
        if not ref_key.startswith(memory_prefix):
            continue

        src_spec = _map_memory_key_to_image_source(ref_key)
        if src_spec is None:
            continue
        src_key, src_index = src_spec
        if src_key not in flat_loaded:
            continue

        src_value = flat_loaded[src_key]
        if src_index is not None:
            if src_value.ndim == 0 or src_index >= src_value.shape[0]:
                continue
            src_value = src_value[src_index]
        if getattr(src_value, "shape", None) != getattr(ref_value, "shape", None):
            continue
        flat_loaded[ref_key] = src_value.astype(ref_value.dtype) if src_value.dtype != ref_value.dtype else src_value


def _map_memory_key_to_image_source(memory_key: str) -> tuple[str, int | None] | None:
    if not memory_key.startswith("PaliGemma/memory_img/"):
        return None

    suffix = memory_key[len("PaliGemma/memory_img/") :]

    direct_mappings = {
        "embedding/kernel": "embedding/kernel",
        "embedding/bias": "embedding/bias",
        "pos_embedding": "pos_embedding",
        "head/kernel": "head/kernel",
        "head/bias": "head/bias",
        "Transformer/encoder_norm/scale": "Transformer/encoder_norm/scale",
        "Transformer/encoder_norm/bias": "Transformer/encoder_norm/bias",
    }
    if suffix in direct_mappings:
        return f"PaliGemma/img/{direct_mappings[suffix]}", None

    if not suffix.startswith("Transformer/encoderblock_"):
        return None

    rest = suffix[len("Transformer/encoderblock_") :]
    block_idx_str, _, block_suffix = rest.partition("/")
    if not block_idx_str.isdigit() or not block_suffix:
        return None

    block_idx = int(block_idx_str)
    img_block_prefix = f"PaliGemma/img/Transformer/encoderblock"

    block_mappings = {
        "spatial_ln/scale": f"{img_block_prefix}/LayerNorm_0/scale",
        "spatial_ln/bias": f"{img_block_prefix}/LayerNorm_0/bias",
        "mlp_ln/scale": f"{img_block_prefix}/LayerNorm_1/scale",
        "mlp_ln/bias": f"{img_block_prefix}/LayerNorm_1/bias",
        "spatial_attn/query/kernel": f"{img_block_prefix}/MultiHeadDotProductAttention_0/query/kernel",
        "spatial_attn/query/bias": f"{img_block_prefix}/MultiHeadDotProductAttention_0/query/bias",
        "spatial_attn/key/kernel": f"{img_block_prefix}/MultiHeadDotProductAttention_0/key/kernel",
        "spatial_attn/key/bias": f"{img_block_prefix}/MultiHeadDotProductAttention_0/key/bias",
        "spatial_attn/value/kernel": f"{img_block_prefix}/MultiHeadDotProductAttention_0/value/kernel",
        "spatial_attn/value/bias": f"{img_block_prefix}/MultiHeadDotProductAttention_0/value/bias",
        "spatial_attn/out/kernel": f"{img_block_prefix}/MultiHeadDotProductAttention_0/out/kernel",
        "spatial_attn/out/bias": f"{img_block_prefix}/MultiHeadDotProductAttention_0/out/bias",
        "mlp/Dense_0/kernel": f"{img_block_prefix}/MlpBlock_0/Dense_0/kernel",
        "mlp/Dense_0/bias": f"{img_block_prefix}/MlpBlock_0/Dense_0/bias",
        "mlp/Dense_1/kernel": f"{img_block_prefix}/MlpBlock_0/Dense_1/kernel",
        "mlp/Dense_1/bias": f"{img_block_prefix}/MlpBlock_0/Dense_1/bias",
    }
    if block_suffix not in block_mappings:
        return None

    return block_mappings[block_suffix], block_idx
