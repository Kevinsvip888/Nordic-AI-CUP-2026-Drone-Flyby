"""CPU-only lifecycle and protocol tests; no model, network or live service."""
from pathlib import Path
import sys
import unittest
import numpy as np
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1] / 'runtime'
sys.path.insert(0, str(ROOT))
from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from utils import encode_image
from live_adapter import LiveKalmanDetector, command_rejection

IMAGE = encode_image(np.zeros((540, 960, 3), dtype=np.uint8))


@dataclass
class Camera:
    resolution_level: int = 0
    center_x: int = 1920
    center_y: int = 1080

    def apply(self, level, x, y):
        self.resolution_level, self.center_x, self.center_y = level, x, y

    @property
    def region(self):
        width, height = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}[self.resolution_level]
        return [self.center_x - width // 2, self.center_y - height // 2,
                self.center_x + width // 2, self.center_y + height // 2]


def build_request(frame, index, camera, encoded):
    allowed = {0: [0, 1], 1: [0, 1, 2], 2: [1, 2]}[camera.resolution_level]
    sizes = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}
    bounds = []
    for level in allowed:
        width, height = sizes[level]
        bounds.append({'resolution_level': level, 'width': 960, 'height': 540,
                       'minimum_center_x': width // 2, 'maximum_center_x': 3840 - width // 2,
                       'minimum_center_y': height // 2, 'maximum_center_y': 2160 - height // 2})
    identity = f'test:{index}'
    return {'sequence_id': 'test', 'frame': frame, 'frame_index': index,
            'request_id': identity, 'frame_interval_ms': 333, 'response_timeout_ms': 3333,
            'original_width': 3840, 'original_height': 2160,
            'view': {'resolution_level': camera.resolution_level, 'center_x': camera.center_x,
                     'center_y': camera.center_y, 'view_id': identity, 'image': encoded,
                     'image_media_type': 'image/png', 'width': 960, 'height': 540,
                     'source_region_xyxy': camera.region},
            'camera_constraints': {'maximum_center_delta': {0: 2203., 1: 1102., 2: 551.}[camera.resolution_level],
                                   'allowed_resolution_levels': allowed, 'center_bounds': bounds,
                                   'full_view_reset_exempt_from_delta': True}}


def request(index=0, sequence='sequence', camera=None):
    data = build_request(index * 10, index, camera or Camera(), IMAGE)
    data.update(sequence_id=sequence, request_id=f'{sequence}:{index}')
    return DroneFlybyPredictRequestDto(**data)


class FakeBase:
    def __init__(self):
        self.calls = 0

    def __call__(self, req):
        self.calls += 1
        left = .2 + min(req.frame_index, 10) * .001
        return DroneFlybyPredictResponseDto(request_id=req.request_id, frame=req.frame, annotations=[
            {'object_id': 'hangar', 'bbox': [left, .2, left + .1, .3], 'confidence': .8},
            {'object_id': 'small_launcher', 'bbox': [.6, .6, .7, .7], 'confidence': .9},
        ])


class BrokenPolicy:
    def __init__(self):
        self.calls = 0
        self.tracks = []

    def step(self, *args):
        self.calls += 1
        return [{'object_id': 'hangar', 'bbox': [0., 0., 0., 1.], 'confidence': .9}], None, {}


class WrapperTests(unittest.TestCase):
    def test_duplicate_and_same_frame_idempotency(self):
        base = FakeBase()
        wrapped = LiveKalmanDetector(base)
        req = request()
        response = wrapped(req)
        self.assertEqual(wrapped(req).model_dump(), response.model_dump())
        alias = req.model_copy(deep=True, update={'request_id': 'other-id'})
        same = wrapped(alias)
        self.assertEqual(same.request_id, 'other-id')
        self.assertEqual([a.model_dump() for a in same.annotations], [a.model_dump() for a in response.annotations])
        self.assertEqual(base.calls, 1)
        self.assertEqual(wrapped.counts['policy_steps'], 1)
        self.assertEqual([a.object_id for a in response.annotations], ['hangar'])

    def test_conflicting_duplicate_rejected_before_base(self):
        base = FakeBase()
        wrapped = LiveKalmanDetector(base)
        req = request()
        wrapped(req)
        changed = req.model_copy(deep=True, update={'frame': 1})
        with self.assertRaises(ValueError):
            wrapped(changed)
        changed.request_id = 'other-request'
        with self.assertRaises(ValueError):
            wrapped(changed)
        self.assertEqual(base.calls, 1)
        self.assertEqual(wrapped.sequences['sequence'].last_index, 0)

    def test_out_of_order_does_not_change_policy_or_cache(self):
        wrapped = LiveKalmanDetector(FakeBase())
        wrapped(request(4))
        state = wrapped.sequences['sequence']
        before = (state.last_index, state.policy.last, list(state.cache), state.policy.next_id)
        response = wrapped(request(2))
        self.assertEqual((state.last_index, state.policy.last, list(state.cache), state.policy.next_id), before)
        self.assertIsNone(response.requested_view)
        self.assertEqual([a.object_id for a in response.annotations], ['hangar'])

    def test_sequence_and_retry_cache_bounds(self):
        wrapped = LiveKalmanDetector(FakeBase(), max_sequences=2, cache_size=2)
        for i in range(4):
            wrapped(request(i))
        self.assertEqual(len(wrapped.sequences['sequence'].cache), 2)
        wrapped(request(0, 'second'))
        wrapped(request(0, 'third'))
        self.assertEqual(list(wrapped.sequences), ['second', 'third'])
        self.assertEqual(wrapped.counts['sequence_evictions'], 1)

    def test_l2_returns_l1_and_clears_policy(self):
        wrapped = LiveKalmanDetector(FakeBase())
        wrapped(request())
        camera = Camera()
        camera.apply(1, 960, 540)
        camera.apply(2, 480, 270)
        req = request(1, camera=camera)
        response = wrapped(req)
        self.assertEqual(response.requested_view.resolution_level, 1)
        self.assertIsNone(command_rejection(req, response.requested_view))
        self.assertEqual(wrapped.sequences['sequence'].policy.tracks, [])
        self.assertEqual(wrapped.counts['unsupported_view_resets'], 1)

    def test_non_aligned_region_returns_l0(self):
        wrapped = LiveKalmanDetector(FakeBase())
        camera = Camera()
        camera.apply(1, 1921, 1081)
        req = request(camera=camera)
        response = wrapped(req)
        self.assertEqual(response.requested_view.resolution_level, 0)
        self.assertIsNone(command_rejection(req, response.requested_view))
        self.assertEqual(wrapped.counts['unsupported_view_resets'], 1)

    def test_dimension_change_resets_and_uses_fresh_outputs(self):
        wrapped = LiveKalmanDetector(FakeBase())
        wrapped(request())
        changed = request(1)
        changed.original_width = 4000
        response = wrapped(changed)
        self.assertEqual(len(response.annotations), 1)
        self.assertEqual(wrapped.counts['dimension_resets'], 1)
        self.assertEqual(wrapped.sequences['sequence'].policy.tracks, [])

    def test_failed_policy_has_no_partial_commit(self):
        wrapped = LiveKalmanDetector(FakeBase())
        wrapped(request())
        state = wrapped.sequences['sequence']
        state.policy = BrokenPolicy()
        before = list(state.cache)
        with self.assertRaises(ValueError):
            wrapped(request(1))
        self.assertEqual(state.policy.calls, 0)
        self.assertEqual(state.last_index, 0)
        self.assertEqual(list(state.cache), before)

    def test_current_observation_updates_position(self):
        wrapped = LiveKalmanDetector(FakeBase())
        first = wrapped(request())
        second = wrapped(request(1))
        self.assertNotEqual(first.annotations[0].bbox, second.annotations[0].bbox)
        self.assertAlmostEqual(second.annotations[0].bbox[0], .201)
        self.assertEqual(wrapped.sequences['sequence'].policy.last, 1)
        self.assertGreater(len(wrapped.snapshot()['diagnostics']), 0)


if __name__ == '__main__':
    unittest.main()
