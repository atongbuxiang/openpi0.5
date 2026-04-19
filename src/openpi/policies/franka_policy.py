import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

FRANKA_STATE_DIM = 8


def make_franka_example() -> dict:
    """Creates a random input example for the Franka policy."""
    return {
        "state": np.ones((FRANKA_STATE_DIM,), dtype=np.float32),
        "images": {
            "base": np.random.randint(256, size=(3, 720, 1280), dtype=np.uint8),
            "wrist": np.random.randint(256, size=(3, 720, 1280), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """Inputs for a Franka LeRobot dataset.

    Expected inputs after repacking:
    - images/base: base camera image
    - images/wrist: wrist camera image
    - state: [7 joint angles, 1 gripper]
    - actions: [action_horizon, 7 joint angles + 1 gripper]
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != FRANKA_STATE_DIM:
            raise ValueError(f"Expected Franka state dim {FRANKA_STATE_DIM}, got {state.shape}")

        in_images = data["images"]
        if "base" not in in_images:
            raise ValueError("Expected Franka images to contain 'base'.")
        if "wrist" not in in_images:
            raise ValueError("Expected Franka images to contain 'wrist'.")

        base_image = _parse_image(in_images["base"])
        wrist_image = _parse_image(in_images["wrist"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                images = {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": wrist_image,
                    "right_wrist_0_rgb": np.zeros_like(base_image),
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.False_,
                }
            case _model.ModelType.PI0_FAST:
                images = {
                    "base_0_rgb": base_image,
                    "base_1_rgb": np.zeros_like(base_image),
                    "wrist_0_rgb": wrist_image,
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "base_1_rgb": np.True_,
                    "wrist_0_rgb": np.True_,
                }
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != FRANKA_STATE_DIM:
                raise ValueError(f"Expected Franka action dim {FRANKA_STATE_DIM}, got {actions.shape}")
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """Outputs for the Franka policy."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Expected Franka actions with shape [T, {FRANKA_STATE_DIM}], got {actions.shape}")
        if actions.shape[-1] < FRANKA_STATE_DIM:
            raise ValueError(f"Expected Franka action dim >= {FRANKA_STATE_DIM}, got {actions.shape}")
        return {"actions": actions[:, :FRANKA_STATE_DIM]}
