"""Image, coordinate and response validation helpers."""

import base64
import math
from typing import Optional, Sequence

import cv2
import numpy as np

from dtos import DroneFlybyPredictResponseDto, View


def decode_image(value: str) -> np.ndarray:
    raw = base64.b64decode(value, validate=True)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Image payload is not a valid encoded image')
    return image


def decode_view(view: View) -> np.ndarray:
    return decode_image(view.image)


def encode_image(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError('Could not encode image')
    return base64.b64encode(encoded.tobytes()).decode('ascii')


def view_bbox_to_global(box: Sequence[float], region: Sequence[int],
                        original_width: int, original_height: int):
    vx1, vy1, vx2, vy2 = map(float, box)
    rx1, ry1, rx2, ry2 = map(float, region)
    width, height = rx2 - rx1, ry2 - ry1
    return (
        (rx1 + vx1 * width) / original_width,
        (ry1 + vy1 * height) / original_height,
        (rx1 + vx2 * width) / original_width,
        (ry1 + vy2 * height) / original_height,
    )


def clip_bbox_to_frame(box: Sequence[float], epsilon: float = 1e-6):
    x1, y1, x2, y2 = map(float, box)
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    return None if x2 - x1 <= epsilon or y2 - y1 <= epsilon else (x1, y1, x2, y2)


def validate_response(response: DroneFlybyPredictResponseDto) -> None:
    if len(response.annotations) > 500:
        raise ValueError('Too many annotations')
    for annotation in response.annotations:
        x1, y1, x2, y2 = map(float, annotation.bbox)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('Invalid normalized bounding box')
        score = float(annotation.confidence)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid confidence')


def describe_camera_rejection(current_level: int, current_center: tuple[int, int],
                              target_level: int, target_center: tuple[int, int]) -> Optional[str]:
    allowed = {0: (0, 1), 1: (0, 1, 2), 2: (1, 2)}
    region_sizes = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}
    movement_limits = {0: 2203.0, 1: 1102.0, 2: 551.0}
    if target_level not in allowed.get(current_level, ()):
        return 'resolution level transition is not allowed'
    width, height = region_sizes[target_level]
    if not width // 2 <= target_center[0] <= 3840 - width // 2:
        return 'camera center_x is outside bounds'
    if not height // 2 <= target_center[1] <= 2160 - height // 2:
        return 'camera center_y is outside bounds'
    if target_level == 0:
        return None if target_center == (1920, 1080) else 'full view must use frame center'
    if math.dist(current_center, target_center) > movement_limits[current_level]:
        return 'camera center movement is too large'
    return None
