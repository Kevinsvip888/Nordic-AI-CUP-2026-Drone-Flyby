"""Map detector outputs in original-view pixels to the official response.

Ultralytics high-level Results.boxes.xyxy is already in original image pixels.
Pass the received 960x540 view to inference and do NOT undo letterboxing again
before using this adapter. Raw network tensor coordinates are not accepted.
This module supplies conversion only; it contains no detector or camera policy.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import math
from numbers import Real

from dtos import (
    OBJECT_CLASSES,
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
)
from utils import clip_bbox_to_frame, validate_response, view_bbox_to_global

logger = logging.getLogger(__name__)


def validate_class_names(names: Mapping[int, str] | Sequence[str]) -> tuple[str, ...]:
    """Reject different class sets/orders, including untouched COCO weights."""
    if isinstance(names, Mapping):
        if any(type(key) is not int for key in names) or set(names) != set(range(len(OBJECT_CLASSES))):
            raise ValueError('Class-name keys must be integers 0 through 15')
        ordered = tuple(names[index] for index in range(len(OBJECT_CLASSES)))
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        ordered = tuple(names)
    else:
        raise ValueError('Class names must be an ordered sequence or integer-keyed mapping')
    if ordered != OBJECT_CLASSES:
        raise ValueError('Class names/order must match dtos.OBJECT_CLASSES exactly')
    return ordered


@dataclass(frozen=True)
class ViewDetection:
    """One postprocessed detection in pixels of the received original view."""

    bbox_xyxy: Sequence[float]
    class_index: int | float
    confidence: float


@dataclass
class ConversionReport:
    annotations: list[DroneFlybyPredictionDto]
    rejected: dict[str, int]
    truncated: int


def _finite_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def convert_detections(
    request: DroneFlybyPredictRequestDto,
    detections: Sequence[ViewDetection],
    names: Mapping[int, str] | Sequence[str],
) -> ConversionReport:
    """Validate, clip to the visible view, map globally, then retain top 500.

    Invalid prediction rows are counted and dropped, never silently repaired
    into a different class/confidence. Class mapping or request geometry errors
    raise immediately because they affect every prediction. No confidence
    threshold or NMS is applied here; those belong to detector configuration.
    """
    ordered_names = validate_class_names(names)
    width, height = request.view.width, request.view.height
    original_width, original_height = request.original_width, request.original_height
    left, top, right, bottom = request.view.source_region_xyxy
    if width <= 0 or height <= 0 or original_width <= 0 or original_height <= 0:
        raise ValueError('View and original image dimensions must be positive')
    if not (0 <= left < right <= original_width and 0 <= top < bottom <= original_height):
        raise ValueError('Source region must be a positive rectangle inside the original frame')

    annotations = []
    rejected = Counter()
    for detection in detections:
        box = detection.bbox_xyxy
        if len(box) != 4 or not all(_finite_number(coordinate) for coordinate in box):
            rejected['invalid_box_numbers'] += 1
            continue
        category = detection.class_index
        if not _finite_number(category) or not float(category).is_integer() or not 0 <= category < len(ordered_names):
            rejected['invalid_class_index'] += 1
            continue
        confidence = detection.confidence
        if not _finite_number(confidence) or not 0 <= confidence <= 1:
            rejected['invalid_confidence'] += 1
            continue
        x1, y1, x2, y2 = map(float, box)
        # Clip in view coordinates first: an overflowing crop prediction must
        # not become an invented detection in unobserved source-frame space.
        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
        if x1 >= x2 or y1 >= y2:
            rejected['degenerate_or_outside_view'] += 1
            continue
        global_box = clip_bbox_to_frame(view_bbox_to_global(
            (x1 / width, y1 / height, x2 / width, y2 / height),
            request.view.source_region_xyxy, original_width, original_height,
        ), epsilon=0)
        if global_box is None:
            rejected['degenerate_global_box'] += 1
            continue
        annotations.append(DroneFlybyPredictionDto(
            object_id=ordered_names[int(category)],
            bbox=list(global_box),
            confidence=float(confidence),
        ))

    annotations.sort(key=lambda annotation: annotation.confidence, reverse=True)
    truncated = max(0, len(annotations) - 500)
    if rejected or truncated:
        logger.warning('Detection conversion dropped %s; truncated %d beyond 500', dict(rejected), truncated)
    return ConversionReport(annotations[:500], dict(rejected), truncated)


def build_response(
    request: DroneFlybyPredictRequestDto,
    detections: Sequence[ViewDetection],
    names: Mapping[int, str] | Sequence[str],
) -> DroneFlybyPredictResponseDto:
    """Echo request identity and hold the current view with requested_view=None."""
    converted = convert_detections(request, detections, names)
    response = DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=converted.annotations,
        requested_view=None,
    )
    validate_response(response)
    return response


def response_from_ultralytics(request: DroneFlybyPredictRequestDto, result) -> DroneFlybyPredictResponseDto:
    """Consume one high-level Ultralytics Results object, without torch imports.

    orig_shape protects against accidentally passing results from a resized
    image. It cannot recognize manually corrupted coordinates; keep Results
    intact from model inference until this boundary.
    """
    validate_class_names(result.names)
    if tuple(result.orig_shape) != (request.view.height, request.view.width):
        raise ValueError('Results.orig_shape must match the received original view')
    if result.boxes is None:
        raise ValueError('Expected detection Results with boxes, not another task type')
    boxes = result.boxes.xyxy.tolist()
    categories = result.boxes.cls.tolist()
    confidences = result.boxes.conf.tolist()
    if not len(boxes) == len(categories) == len(confidences):
        raise ValueError('Detector box, class, and confidence arrays must have equal length')
    detections = [ViewDetection(box, category, confidence)
                  for box, category, confidence in zip(boxes, categories, confidences)]
    return build_response(request, detections, result.names)
