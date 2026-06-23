"""Central user-configurable parameters for the cone-track task.

Both env.py and actor_critic_vision.py import from here so that changes
propagate automatically.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Camera / vision pipeline
# ═══════════════════════════════════════════════════════════════════════════

CAM_H: int = 480
CAM_W: int = 848

VISION_CROP_TOP_FRACTION: float = 0.5
VISION_CROP_BOTTOM_FRACTION: float = 0.0

DOWNSAMPLING_AMOUNT: int = 1

INCLUDE_WHEEL_VEL: bool = True

# Derived policy-image dimensions (computed once here)
_remaining = max(0.0, 1.0 - VISION_CROP_TOP_FRACTION - VISION_CROP_BOTTOM_FRACTION)
_cropped_h = max(1, int(CAM_H * _remaining))
_cropped_w = CAM_W

POLICY_DEPTH_W: int = max(1, _cropped_w // DOWNSAMPLING_AMOUNT)
POLICY_DEPTH_H: int = max(1, _cropped_h // DOWNSAMPLING_AMOUNT)
VISION_OBS_SIZE: int = POLICY_DEPTH_H * POLICY_DEPTH_W

# ═══════════════════════════════════════════════════════════════════════════
# CNN encoder architecture
# ═══════════════════════════════════════════════════════════════════════════

CNN_CHANNELS: tuple = (32, 64, 128)
CNN_KERNEL_SIZES: tuple = (5, 5, 3)
CNN_STRIDES: tuple = (2, 2, 2)
CNN_POOL_SIZE: tuple = (3, 3)
