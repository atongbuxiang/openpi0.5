import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_lerobot_example() -> dict:
    """Creates a random input example for the LeRobot policy."""
    return {
        "observation/state": np.ones((14,)),
        "observation/images": {
            "cam_high": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
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
class LeRobotInputs(transforms.DataTransformFn):
    """Inputs for the LeRobot policy.

    Expected inputs:
    - observation/images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - observation/state: [14] joint and gripper positions
    - actions: [action_horizon, 14]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST.
        mask_padding = self.model_type == _model.ModelType.PI0 or self.model_type == _model.ModelType.PI05

        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Get the images
        in_images = data["observation/images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Process the base image (cam_high)
        base_image = _parse_image(in_images["cam_high"])

        # Process wrist images
        left_wrist_image = _parse_image(in_images["cam_left_wrist"]) if "cam_left_wrist" in in_images else np.zeros_like(base_image)
        right_wrist_image = _parse_image(in_images["cam_right_wrist"]) if "cam_right_wrist" in in_images else np.zeros_like(base_image)

        # Create inputs dict. The keys in this dict match what the model expects.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if "cam_left_wrist" in in_images else np.False_ if mask_padding else np.True_,
                "right_wrist_0_rgb": np.True_ if "cam_right_wrist" in in_images else np.False_ if mask_padding else np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (language instruction) to the model if available.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LeRobotOutputs(transforms.DataTransformFn):
    """Outputs for the LeRobot policy."""

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims of actions, which matches the action space of the LeRobot
        # (as defined in info.json)
        return {"actions": np.asarray(data["actions"][:, :14])}
