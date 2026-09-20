"""Serve a verified B0 package; default binding is loopback, one GPU process."""

import argparse
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import threading
import time

from fastapi import FastAPI
import uvicorn

from baseline_endpoint import INFERENCE_CONFIG, TrainedDetector
from dtos import OBJECT_CLASSES, DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from utils import validate_response, describe_camera_rejection
from request_recorder import RequestRecorder

LOGGER = logging.getLogger('b0.service')
POLICY_REFERENCE = None
REQUIRED_FILES = {'serve_b0.py', 'baseline_endpoint.py', 'detector_adapter.py',
                  'dtos.py', 'utils.py', 'model.pt', 'request_recorder.py'}


def emit(event, **fields):
    LOGGER.info(json.dumps({'event': event, **fields}, allow_nan=False))


def verify_package(manifest_path, *, check_runtime=True):
    path = Path(manifest_path).resolve(strict=True)
    root = path.parent
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['schema_version'] != 1 or manifest['inference'] != INFERENCE_CONFIG:
        raise ValueError('Unsupported package or changed B0 inference configuration')
    if manifest['class_names'] != list(OBJECT_CLASSES):
        raise ValueError('Package class order mismatch')
    if manifest['checkpoint'] != 'model.pt' or not REQUIRED_FILES <= set(manifest['files']):
        raise ValueError('Package is missing required serving files')
    for name, expected in manifest['files'].items():
        candidate = (root / name).resolve(strict=True)
        if not candidate.is_relative_to(root):
            raise ValueError('Package paths must stay inside the package')
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Package file hash mismatch: {name}')
    if check_runtime:
        if platform.python_version_tuple()[:2] != ('3', '12'):
            raise RuntimeError('This package requires the tested Python 3.12 runtime')
        for name, expected in manifest['runtime_versions'].items():
            if importlib.metadata.version(name) != expected:
                raise RuntimeError(f'Runtime version mismatch for {name}; expected {expected}')
    return manifest, root / manifest['checkpoint']


def load_detector(manifest_path):
    manifest, checkpoint = verify_package(manifest_path)
    started = time.perf_counter()
    detector = TrainedDetector(checkpoint)
    if not detector.runtime['startup_warmup']['completed']:
        raise RuntimeError('Detector warm-up did not complete')
    metadata = {
        'checkpoint_sha256': detector.checkpoint_sha256,
        'manifest_sha256': hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        'inference': INFERENCE_CONFIG, 'runtime': detector.runtime,
        'runtime_versions': manifest['runtime_versions'],
        'startup_seconds': time.perf_counter() - started,
    }
    emit('ready', **metadata)
    return detector, metadata


def create_service(detector, metadata, recorder=None):
    if not metadata['runtime']['startup_warmup']['completed']:
        raise RuntimeError('Cannot serve before warm-up completes')
    @asynccontextmanager
    async def lifespan(app):
        yield
        if recorder is not None:
            # Inference has stopped; allow a bounded drain on normal shutdown.
            closed = recorder.close(timeout=10)
            emit('recording_shutdown', writer_stopped=closed, **recorder.snapshot())

    app = FastAPI(title='Frozen B0 detector', lifespan=lifespan)
    lock = threading.Lock()
    counts = {'requests': 0, 'fallbacks': 0, 'recording_submit_errors': 0}

    @app.get('/')
    @app.get('/ready')
    def ready():
        with lock:
            return {'ready': True, 'camera_policy': metadata['camera_policy'], 'policy_snapshot': POLICY_REFERENCE.snapshot() if POLICY_REFERENCE else None, 'tta': metadata.get('tta'), 'class_alternatives': metadata.get('class_alternatives'), 'actual_fp16': metadata.get('actual_fp16'), 'persistent_inference_worker': metadata.get('persistent_inference_worker'),
                    'checkpoint_sha256': metadata['checkpoint_sha256'], **counts,
                    'recording': recorder.snapshot() if recorder else {'enabled': False}}

    @app.post('/predict', response_model=DroneFlybyPredictResponseDto)
    def predict(request: DroneFlybyPredictRequestDto):
        started = time.perf_counter()
        received_utc = datetime.now(timezone.utc).isoformat()
        status, error_type = 'ok', None
        try:
            response = detector(request)
            # Revalidate a serialized copy: model instances can have been mutated.
            response = DroneFlybyPredictResponseDto.model_validate(response.model_dump())
            validate_response(response)
            if response.request_id != request.request_id or response.frame != request.frame:
                raise ValueError('Detector response identity mismatch')
            if response.requested_view is not None:
                reason = describe_camera_rejection(request.view.resolution_level, (request.view.center_x, request.view.center_y), response.requested_view.resolution_level, (response.requested_view.center_x, response.requested_view.center_y))
                if reason: raise ValueError(reason)
        except Exception as exc:
            # Startup failures never enter this handler. Per-request failures
            # lose detections but must not poison the wire format or next call.
            status, error_type = 'empty_fallback', type(exc).__name__
            response = DroneFlybyPredictResponseDto(
                request_id=request.request_id, frame=request.frame,
                annotations=[], requested_view=None)
        with lock:
            counts['requests'] += 1
            counts['fallbacks'] += int(status != 'ok')
        recording_queued = None
        if recorder is not None:
            try:
                recording_queued = recorder.submit(request, response,
                    received_utc=received_utc, pre_recording_handler_ms=(time.perf_counter() - started) * 1000,
                    status=status, error_type=error_type)
            except Exception as exc:
                # Recorder implementation failures cannot take down inference.
                recording_queued = False
                with lock:
                    counts['recording_submit_errors'] += 1
                recorder.note_submission_error(type(exc).__name__)
                emit('recording_submit_error', request_id=request.request_id, error_type=type(exc).__name__)
        emit('request', sequence_id=request.sequence_id, request_id=request.request_id,
             frame=request.frame, frame_index=request.frame_index,
             level=request.view.resolution_level, status=status, error_type=error_type,
             detections=len(response.annotations),
             recording_queued=recording_queued,
             handler_ms=(time.perf_counter() - started) * 1000)
        return response

    return app


def main():
    global POLICY_REFERENCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).with_name('b0_package.json'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9054)
    parser.add_argument('--log-file', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--record-dir', type=Path, help='New directory for this server session; recording is off by default')
    parser.add_argument('--record-queue', type=int, default=16)
    parser.add_argument('--record-max-mib', type=int, default=2048)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding='utf-8'))
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers = handlers
    LOGGER.propagate = False
    if args.verify_only:
        manifest, _ = verify_package(args.manifest)
        emit('verified', checkpoint_sha256=manifest['files']['model.pt'])
        return
    from inference_worker import InferenceWorker
    worker = InferenceWorker()
    detector, metadata = worker.run(load_detector, args.manifest)
    from multilabel_tta import MultiLabelTTA
    detector = worker.run(MultiLabelTTA, detector, 'top2')
    metadata['persistent_inference_worker'] = True
    metadata['class_alternatives'] = 'top2'
    metadata['actual_fp16'] = detector.bases[0].model.predictor.model.fp16
    assert metadata['actual_fp16'] is True
    from live_adapter import LiveKalmanDetector
    detector = worker.run(LiveKalmanDetector, detector)
    POLICY_REFERENCE = detector
    metadata['camera_policy'] = 'L0-L1-L0 only; mandatory immediate return after every L1'
    metadata['memory_policy'] = detector.configuration()
    detector = worker.wrap(detector)
    metadata['tta'] = {'mode':'wbf','views':['original','h','v','hv'],'merge_iou':.55,'missing_view_confidence':0}
    recorder = (RequestRecorder(args.record_dir, metadata, capacity=args.record_queue,
                max_total_bytes=args.record_max_mib * 1024 ** 2) if args.record_dir else None)
    # Model loading and synchronized warm-up precede binding the listening socket.
    try:
        uvicorn.run(create_service(detector, metadata, recorder), host=args.host, port=args.port,
                    workers=1, access_log=False)
    finally:
        worker.close()


if __name__ == '__main__':
    main()
