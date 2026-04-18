from collections import deque
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable


import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data

@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        # q01, q99 = stats.q01, stats.q99
        x = np.array(x)
        q01 = np.array(stats.q01)
        q99 = np.array(stats.q99)
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class StackHistory(DataTransformFn):
    """Stacks current and historical image keys into `[T, H, W, C]`.

    The input is expected to be in the common `"image"` / `"image_mask"` format after
    dataset-specific transforms have run.
    """

    num_frames: int
    image_keys: Sequence[str] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if self.num_frames <= 1:
            return data
        if "image" not in data:
            return data

        image_keys = tuple(self.image_keys or tuple(data["image"].keys()))
        output_images = dict(data["image"])
        output_masks = dict(data.get("image_mask", {}))

        for key in image_keys:
            image = np.asarray(output_images[key])
            mask = np.asarray(output_masks.get(key, np.True_))
            if image.ndim == 3:
                image = np.repeat(image[None, ...], self.num_frames, axis=0)
                mask = np.repeat(np.asarray(mask)[None, ...], self.num_frames, axis=0)
            elif image.ndim != 4:
                raise ValueError(f"Expected image {key} to have ndim 3 or 4, got {image.shape}")

            if image.shape[0] != self.num_frames:
                if image.shape[0] > self.num_frames:
                    image = image[-self.num_frames :]
                    mask = np.asarray(mask)[-self.num_frames :]
                else:
                    pad_count = self.num_frames - image.shape[0]
                    image_pad = np.repeat(image[:1], pad_count, axis=0)
                    mask_pad = np.zeros((pad_count,), dtype=np.bool_)
                    image = np.concatenate([image_pad, image], axis=0)
                    mask = np.concatenate([mask_pad, np.asarray(mask)], axis=0)

            output_images[key] = image
            output_masks[key] = np.asarray(mask, dtype=np.bool_)

        return {**data, "image": output_images, "image_mask": output_masks}


@dataclasses.dataclass
class OnlineHistoryBuffer(DataTransformFn):
    """Stateful history buffer for policy inference.

    Converts a stream of single-frame images into `[T, H, W, C]` short videos.
    """

    num_frames: int
    image_keys: Sequence[str] | None = None

    def __post_init__(self):
        self._buffers = {}

    def __call__(self, data: DataDict) -> DataDict:
        if self.num_frames <= 1 or "image" not in data:
            return data

        image_keys = tuple(self.image_keys or tuple(data["image"].keys()))
        output_images = dict(data["image"])
        output_masks = dict(data.get("image_mask", {}))

        for key in image_keys:
            image = np.asarray(output_images[key])
            if image.ndim == 4:
                output_images[key] = image
                mask = np.asarray(output_masks.get(key, np.ones((image.shape[0],), dtype=np.bool_)), dtype=np.bool_)
                output_masks[key] = mask
                self._buffers[key] = deque([frame.copy() for frame in image], maxlen=self.num_frames)
                continue
            if image.ndim != 3:
                raise ValueError(f"Expected image {key} to have ndim 3 or 4, got {image.shape}")

            if key not in self._buffers:
                self._buffers[key] = deque(maxlen=self.num_frames)
            self._buffers[key].append(image.copy())

            frames = list(self._buffers[key])
            pad = self.num_frames - len(frames)
            if pad > 0:
                zero = np.zeros_like(image)
                stacked = [zero for _ in range(pad)] + frames
                mask = np.array([False] * pad + [True] * len(frames), dtype=np.bool_)
            else:
                stacked = frames
                mask = np.ones((self.num_frames,), dtype=np.bool_)

            output_images[key] = np.stack(stacked, axis=0)
            output_masks[key] = mask

        return {**data, "image": output_images, "image_mask": output_masks}


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}

@dataclasses.dataclass(frozen=True)
class PromptFromMultiLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current MultiLeRobotDataset task."""

    tasks: dict[str, str]  # 注意：这里的 key 是 "repo_id_taskIndex"
    repo_ids: list[str]    # 新增字段，存储所有 repo_id

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')
        if "dataset_index" not in data:
            raise ValueError('Cannot extract repo_id without "dataset_index"')

        task_index = int(data["task_index"])
        dataset_index = int(data["dataset_index"])
        repo_id = self.repo_ids[dataset_index]  
        # print(f"Debug: dataset_index={dataset_index}, repo_id={repo_id}")
        
        # 构造 tasks 中的 key
        task_key = f"{repo_id}_{task_index}"
        if (prompt := self.tasks.get(task_key)) is None:
            raise ValueError(f"{task_key} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt, "repo_id": repo_id}

@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )



@dataclasses.dataclass(frozen=True)
class PackDualArmJointGripperState:
    """Pack left/right joint positions + gripper scalar into a flat observation.state vector.

    Output layout:
      [left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]  -> (14,)
    """

    left_joint_key: str = "observation.state.arm.left.joint_positions"
    left_gripper_key: str = "observation.state.arm.left.end_effector_value"
    right_joint_key: str = "observation.state.arm.right.joint_positions"
    right_gripper_key: str = "observation.state.arm.right.end_effector_value"
    out_key: str = "__packed_state"
    dtype: Any = np.float32
    drop_source_keys: bool = False

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        def _as_1d(a: Any) -> np.ndarray:
            arr = np.asarray(a, dtype=self.dtype)
            if arr.ndim >= 2:
                arr = arr[0]
            return arr.reshape(-1)

        def _as_scalar(a: Any) -> np.ndarray:
            arr = np.asarray(a, dtype=self.dtype).reshape(-1)
            # 若是序列，取第一个元素
            if arr.size > 1:
                arr = arr[:1]
            return arr

        lj = _as_1d(x[self.left_joint_key])
        rj = _as_1d(x[self.right_joint_key])

        lg = _as_scalar(x[self.left_gripper_key])  # 保证是(1,)
        rg = _as_scalar(x[self.right_gripper_key])

        state = np.concatenate([lj, lg, rj, rg], axis=0)

        y = dict(x)
        y[self.out_key] = state

        if self.drop_source_keys:
            for k in (self.left_joint_key, self.left_gripper_key, self.right_joint_key, self.right_gripper_key):
                y.pop(k, None)

        return y


@dataclasses.dataclass(frozen=True)
class PackDualArmJointGripperActionSequence:
    """Pack left/right joint positions + gripper scalar into a flat actions sequence.

    Output layout per timestep:
      [left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]  -> (14,)
    """

    left_joint_key: str = "observation.state.arm.left.joint_positions"
    left_gripper_key: str = "observation.state.arm.left.end_effector_value"
    right_joint_key: str = "observation.state.arm.right.joint_positions"
    right_gripper_key: str = "observation.state.arm.right.end_effector_value"
    out_key: str = "__packed_actions"
    dtype: Any = np.float32
    drop_source_keys: bool = False

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        def _as_2d(a: Any) -> np.ndarray:
            arr = np.asarray(a, dtype=self.dtype)
            if arr.ndim == 1:
                arr = arr[None, :]
            return arr

        lj = _as_2d(x[self.left_joint_key])
        rj = _as_2d(x[self.right_joint_key])

        lg = _as_2d(x[self.left_gripper_key]).reshape(lj.shape[0], -1)
        rg = _as_2d(x[self.right_gripper_key]).reshape(rj.shape[0], -1)

        actions = np.concatenate([lj, lg, rj, rg], axis=-1)

        y = dict(x)
        y[self.out_key] = actions

        if self.drop_source_keys:
            for k in (self.left_joint_key, self.left_gripper_key, self.right_joint_key, self.right_gripper_key):
                y.pop(k, None)

        return y
