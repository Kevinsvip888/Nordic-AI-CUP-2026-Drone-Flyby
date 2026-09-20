"""Strict data models for the detector request and response protocol."""

import math
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, StrictBool, StrictFloat, StrictInt
from pydantic import conint, conlist, field_validator

IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160
TRANSMITTED_VIEW_SIZE = (960, 540)
SOURCE_REGION_SIZES = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}
OBJECT_CLASSES = (
    'hangar', 'helicopter', 'jet_plane', 'large_launcher', 'large_tower',
    'medium_launcher', 'medium_plane', 'mine_roller', 'small_launcher',
    'small_plane', 'small_tower', 'ta-ta', 'tank', 'condor', 'jammer',
    'spacecraft',
)

Number = Union[StrictInt, StrictFloat]
FrameNumber = conint(strict=True, ge=0)
ResolutionLevel = conint(strict=True, ge=0, le=2)
NormalizedBox = conlist(Number, min_length=4, max_length=4)
PixelBox = conlist(StrictInt, min_length=4, max_length=4)


class View(BaseModel):
    resolution_level: ResolutionLevel
    center_x: StrictInt
    center_y: StrictInt
    view_id: str
    image: str
    image_media_type: str
    width: StrictInt
    height: StrictInt
    source_region_xyxy: PixelBox
    model_config = ConfigDict(extra='ignore')


class LevelBounds(BaseModel):
    resolution_level: ResolutionLevel
    width: StrictInt
    height: StrictInt
    minimum_center_x: StrictInt
    maximum_center_x: StrictInt
    minimum_center_y: StrictInt
    maximum_center_y: StrictInt
    model_config = ConfigDict(extra='ignore')


class CameraConstraints(BaseModel):
    maximum_center_delta: float
    allowed_resolution_levels: list[ResolutionLevel]
    center_bounds: list[LevelBounds]
    full_view_reset_exempt_from_delta: StrictBool
    model_config = ConfigDict(extra='ignore')

    def bounds_for_level(self, level: int) -> Optional[LevelBounds]:
        return next((item for item in self.center_bounds
                     if item.resolution_level == level), None)


class RequestedViewDto(BaseModel):
    resolution_level: StrictInt
    center_x: StrictInt
    center_y: StrictInt
    model_config = ConfigDict(extra='forbid', frozen=True)


class CameraFeedback(BaseModel):
    frame: FrameNumber
    requested_view: RequestedViewDto
    reason: str
    model_config = ConfigDict(extra='ignore')


class DroneFlybyPredictRequestDto(BaseModel):
    sequence_id: str
    frame: FrameNumber
    frame_index: FrameNumber
    request_id: str
    frame_interval_ms: conint(strict=True, ge=0)
    response_timeout_ms: conint(strict=True, ge=0)
    original_width: StrictInt
    original_height: StrictInt
    view: View
    camera_constraints: CameraConstraints
    camera_command_feedback: Optional[CameraFeedback] = None
    model_config = ConfigDict(extra='ignore')


class DroneFlybyPredictionDto(BaseModel):
    object_id: str
    bbox: NormalizedBox
    confidence: Number
    model_config = ConfigDict(extra='forbid')

    @field_validator('object_id')
    @classmethod
    def known_class(cls, value: str) -> str:
        if value not in OBJECT_CLASSES:
            raise ValueError('Unknown object class')
        return value

    @field_validator('bbox')
    @classmethod
    def valid_box(cls, value):
        x1, y1, x2, y2 = map(float, value)
        if not all(map(math.isfinite, (x1, y1, x2, y2))):
            raise ValueError('Bounding box values must be finite')
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('Bounding box must be normalized xyxy')
        return value

    @field_validator('confidence')
    @classmethod
    def valid_confidence(cls, value):
        score = float(value)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Confidence must be finite and within [0, 1]')
        return value


class DroneFlybyPredictResponseDto(BaseModel):
    request_id: str
    frame: FrameNumber
    annotations: conlist(DroneFlybyPredictionDto, max_length=500)
    requested_view: Optional[RequestedViewDto] = None
    model_config = ConfigDict(extra='forbid')
