import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_pnd_example() -> dict:
    """Creates a random input example for the PND policy."""
    return {
        "state": np.ones((21,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 540, 960), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class PNDInputs(transforms.DataTransformFn):
    """Inputs for the PND policy.

    Assumptions:
    - state dim = 21
    - action dim = 21
    - first 19 dims are motor joints
    - last 2 dims are grippers
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high",)

    def __call__(self, data: dict) -> dict:
        data = _decode_pnd(data)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain only {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        if "cam_high" not in in_images:
            raise ValueError("Expected images to contain 'cam_high'.")

        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.ones((base_image.shape[0],), dtype=np.bool_) if base_image.ndim == 4 else np.True_,
        }

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.ones((images[dest].shape[0],), dtype=np.bool_) if images[dest].ndim == 4 else np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.zeros((images[dest].shape[0],), dtype=np.bool_) if images[dest].ndim == 4 else np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 21:
                raise ValueError(f"Expected action dim 21, got {actions.shape}")
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PNDOutputs(transforms.DataTransformFn):
    """Outputs for the PND policy."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)

        if actions.ndim != 2:
            raise ValueError(f"Expected actions with shape [T, 21], got {actions.shape}")
        if actions.shape[-1] < 21:
            raise ValueError(f"Expected action dim >= 21, got {actions.shape}")

        # Return first 21 dims for PND:
        # first 19 dims = motors, last 2 dims = grippers
        actions = actions[:, :21]
        return {"actions": actions}


def _decode_pnd(data: dict) -> dict:
    state = np.asarray(data["state"], dtype=np.float32)
    if state.shape[-1] != 21:
        raise ValueError(f"Expected state dim 21, got {state.shape}")

    def convert_image(img):
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        if img.ndim == 3 and img.shape[0] == 3:
            return einops.rearrange(img, "c h w -> h w c")
        if img.ndim == 4 and img.shape[1] == 3:
            return einops.rearrange(img, "t c h w -> t h w c")
        return img

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    out = dict(data)
    out["images"] = images_dict
    out["state"] = state
    return out
