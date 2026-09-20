"""Transactional, bounded per-sequence adapter for the frozen Kalman policy.

The caller owns detector construction and inference-thread affinity. This module
does not load model weights, write files, or make network requests.
"""
from collections import Counter, OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import threading

from dtos import DroneFlybyPredictResponseDto, RequestedViewDto
from utils import decode_view, validate_response
from camera_policy import MemoryCameraPolicy, EXCLUDED


@dataclass
class SequenceState:
    dimensions: tuple
    policy: object
    last_index: int | None = None
    cache: OrderedDict = field(default_factory=OrderedDict)


def _clean(annotation):
    if hasattr(annotation, 'model_dump'):
        annotation = annotation.model_dump()
    return {key: deepcopy(annotation[key]) for key in ('object_id', 'bbox', 'confidence')}


def command_rejection(request, command):
    """Use constraints from the actual incoming request, including reset policy."""
    if command is None:
        return None
    if not isinstance(command, RequestedViewDto):
        command = RequestedViewDto.model_validate(command)
    constraints = request.camera_constraints
    if command.resolution_level not in constraints.allowed_resolution_levels:
        return 'resolution_level_not_allowed'
    bounds = constraints.bounds_for_level(command.resolution_level)
    if bounds is None:
        return 'missing_target_bounds'
    if not bounds.minimum_center_x <= command.center_x <= bounds.maximum_center_x:
        return 'center_x_outside_bounds'
    if not bounds.minimum_center_y <= command.center_y <= bounds.maximum_center_y:
        return 'center_y_outside_bounds'
    exempt = command.resolution_level == 0 and constraints.full_view_reset_exempt_from_delta
    distance = math.hypot(command.center_x - request.view.center_x,
                          command.center_y - request.view.center_y)
    if not exempt and distance > constraints.maximum_center_delta + 1e-9:
        return 'center_movement_too_large'
    return None


def return_toward_l0(request):
    """Return L1 to L0 and L2 to L1 when legal; never jump L2 to L0."""
    current = request.view.resolution_level
    if current == 0:
        return None
    target = current - 1
    bounds = request.camera_constraints.bounds_for_level(target)
    if bounds is None or target not in request.camera_constraints.allowed_resolution_levels:
        return None
    x = min(bounds.maximum_center_x, max(bounds.minimum_center_x, request.view.center_x))
    y = min(bounds.maximum_center_y, max(bounds.minimum_center_y, request.view.center_y))
    command = {'resolution_level': int(target), 'center_x': int(x), 'center_y': int(y)}
    return command if command_rejection(request, command) is None else None


def unsupported_reason(request, image):
    if (request.original_width, request.original_height) != (3840, 2160):
        return 'unsupported_source_dimensions'
    if (request.view.width, request.view.height) != (960, 540) or image.shape != (540, 960, 3):
        return 'unsupported_view_dimensions'
    level = request.view.resolution_level
    if level not in (0, 1):
        return 'unsupported_level_return_via_L1'
    region = request.view.source_region_xyxy
    if any(value % 4 for value in region):
        return 'source_region_not_aligned_to_four'
    x1, y1, x2, y2 = region
    expected_size = (3840, 2160) if level == 0 else (1920, 1080)
    if not (0 <= x1 < x2 <= 3840 and 0 <= y1 < y2 <= 2160):
        return 'source_region_outside_frame'
    if (x2 - x1, y2 - y1) != expected_size:
        return 'source_region_size_mismatch'
    if (x1 + x2, y1 + y2) != (request.view.center_x * 2, request.view.center_y * 2):
        return 'source_region_center_mismatch'
    return None


class LiveKalmanDetector:
    def __init__(self, base, *, max_sequences=8, cache_size=32):
        if not 1 <= max_sequences <= 8 or not 1 <= cache_size <= 32:
            raise ValueError('State limits must be within 8 sequences and 32 cached requests per sequence')
        self.base = base
        self.max_sequences = max_sequences
        self.cache_size = cache_size
        self.sequences = OrderedDict()
        self.diagnostics = deque(maxlen=512)
        self.counts = Counter()
        self.lock = threading.RLock()

    def configuration(self):
        return {'policy': MemoryCameraPolicy().configuration(), 'max_sequences': self.max_sequences,
                'cached_requests_per_sequence': self.cache_size, 'diagnostic_capacity': 512,
                'state_commit': 'after response and command validation',
                'actual_incoming_view_authoritative': True}

    def snapshot(self, limit=32):
        with self.lock:
            limit = max(0, min(512, int(limit)))
            return {
                'camera_policy': 'YOLO L0 E50 with per-track Kalman memory; three L0 views and class/confidence-change camera v6',
                'counts': dict(self.counts), 'active_sequences': len(self.sequences),
                'sequences': [{'sequence_id': key, 'last_frame_index': state.last_index,
                               'dimensions': list(state.dimensions), 'cached_requests': len(state.cache),
                               'tracks': len(state.policy.tracks)} for key, state in self.sequences.items()],
                'diagnostics': deepcopy(list(self.diagnostics)[-limit:]) if limit else [],
            }

    @staticmethod
    def _fingerprint(request):
        value = request.model_dump(mode='json')
        value.pop('request_id', None)
        # Do not retain image payloads in the cache or diagnostics.
        value['view']['image'] = hashlib.sha256(value['view']['image'].encode('ascii')).hexdigest()
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode('utf8')).hexdigest()

    def _record(self, request, status, **fields):
        self.diagnostics.append({'sequence_id': request.sequence_id, 'request_id': request.request_id,
            'frame': request.frame, 'frame_index': request.frame_index,
            'view_level': request.view.resolution_level, 'source_region': list(request.view.source_region_xyxy),
            'status': status, **deepcopy(fields)})

    def _cache(self, state, request, fingerprint, response):
        state.cache[request.request_id] = {'fingerprint': fingerprint,
            'frame_index': request.frame_index, 'response': deepcopy(response.model_dump())}
        state.cache.move_to_end(request.request_id)
        while len(state.cache) > self.cache_size:
            state.cache.popitem(last=False)

    def _checked_response(self, request, annotations, command):
        response = DroneFlybyPredictResponseDto(request_id=request.request_id, frame=request.frame,
            annotations=[_clean(annotation) for annotation in annotations], requested_view=command)
        validate_response(response)
        reason = command_rejection(request, response.requested_view)
        if reason:
            raise ValueError('Unsafe camera command: ' + reason)
        return response

    def _cached_response(self, request, state, cached, fingerprint, *, same_frame=False):
        data = deepcopy(cached['response'])
        data['request_id'] = request.request_id
        response = DroneFlybyPredictResponseDto.model_validate(data)
        validate_response(response)
        assert response.frame == request.frame
        assert command_rejection(request, response.requested_view) is None
        self._cache(state, request, fingerprint, response)
        self.sequences.move_to_end(request.sequence_id)
        self.counts['same_frame_cache_hits' if same_frame else 'cache_hits'] += 1
        self._record(request, 'same_frame_cached' if same_frame else 'cached')
        return response

    def __call__(self, request):
        # Serialize state and the underlying detector, including retries. The
        # service can run this callable within its existing GPU worker.
        with self.lock:
            self.counts['requests'] += 1
            try:
                return self._predict(request)
            except Exception as error:
                self.counts['errors'] += 1
                self._record(request, 'error_no_policy_commit', error_type=type(error).__name__)
                raise

    def _predict(self, request):
        key = request.sequence_id
        fingerprint = self._fingerprint(request)
        dimensions = (request.original_width, request.original_height, request.view.width, request.view.height)
        original = self.sequences.get(key)
        if original is not None and request.request_id in original.cache:
            cached = original.cache[request.request_id]
            if cached['fingerprint'] != fingerprint:
                self.counts['duplicate_conflicts'] += 1
                raise ValueError('Conflicting duplicate request_id in sequence')
            return self._cached_response(request, original, cached, fingerprint)
        state = original
        reset_dimensions = state is not None and state.dimensions != dimensions
        if state is None or reset_dimensions:
            state = SequenceState(dimensions=dimensions, policy=MemoryCameraPolicy())
        else:
            for cached in state.cache.values():
                if cached['frame_index'] != request.frame_index:
                    continue
                if cached['fingerprint'] != fingerprint:
                    self.counts['duplicate_conflicts'] += 1
                    raise ValueError('Conflicting duplicate frame_index in sequence')
                return self._cached_response(request, state, cached, fingerprint, same_frame=True)
        # Inference occurs only after duplicate checks. The base detector must
        # return source-normalized fresh detections for the actual incoming view.
        raw_response = self.base(request)
        raw_response = DroneFlybyPredictResponseDto.model_validate(raw_response.model_dump())
        validate_response(raw_response)
        if raw_response.request_id != request.request_id or raw_response.frame != request.frame:
            raise ValueError('Underlying detector response identity mismatch')
        raw = [_clean(annotation) for annotation in raw_response.annotations]
        fresh = [annotation for annotation in raw if annotation['object_id'] not in EXCLUDED]
        if state.last_index is not None and request.frame_index <= state.last_index:
            # Evicted historical requests cannot advance or rewind the tracker.
            # They deliberately do not mutate state/cache/LRU, only diagnostics.
            command = return_toward_l0(request) if request.view.resolution_level == 1 else None
            response = self._checked_response(request, fresh, command)
            self.counts['out_of_order_fresh_only'] += 1
            self._record(request, 'out_of_order_fresh_only', state_last_index=state.last_index,
                         requested_view=command)
            return response
        observed = decode_view(request.view)
        reason = unsupported_reason(request, observed)
        # All mutable policy work is staged. An exception or invalid DTO cannot
        # leave partially advanced tracks, motion buffers, or cooldowns behind.
        staged = MemoryCameraPolicy() if reason else deepcopy(state.policy)
        if reason:
            outputs = fresh
            command = return_toward_l0(request)
            diagnostic = {'reason': reason, 'decision': 'fresh_only_return_toward_L0',
                          'events': [], 'observed_raw_count': len(raw)}
        else:
            outputs, command, diagnostic = staged.step(request.frame_index,
                request.view.resolution_level, list(request.view.source_region_xyxy), raw, observed)
        rejection = command_rejection(request, command)
        if rejection:
            command = return_toward_l0(request)
            diagnostic = {**diagnostic, 'command_rejected_locally': rejection,
                          'command_substituted_with_legal_return': command}
        response = self._checked_response(request, outputs, command)
        # Commit after all validation. State is keyed by actual sequence, never
        # by a requested camera view; the next request supplies authoritative view.
        state.policy = staged
        state.last_index = request.frame_index
        self._cache(state, request, fingerprint, response)
        self.sequences[key] = state
        self.sequences.move_to_end(key)
        if original is None:
            self.counts['sequences_created'] += 1
        if reset_dimensions:
            self.counts['dimension_resets'] += 1
        if reason:
            self.counts['unsupported_view_resets'] += 1
        else:
            self.counts['policy_steps'] += 1
        if rejection:
            self.counts['illegal_commands_replaced'] += 1
        while len(self.sequences) > self.max_sequences:
            self.sequences.popitem(last=False)
            self.counts['sequence_evictions'] += 1
        self.counts['responses_committed'] += 1
        self.counts['memory_boxes'] += sum(annotation.get('kind') == 'memory_prediction' for annotation in outputs)
        self._record(request, 'unsupported_view_fresh_only' if reason else 'policy_committed',
            requested_view=command, diagnostic=diagnostic,
            output_count=len(response.annotations), camera_command_feedback=(
                request.camera_command_feedback.model_dump() if request.camera_command_feedback else None))
        return response
