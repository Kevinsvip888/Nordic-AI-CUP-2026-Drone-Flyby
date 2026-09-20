"""Local-only trained detector endpoint; never binds to an external interface.

Run: python baseline_endpoint.py --checkpoint artifacts/baseline/.../best.pt
The unchanged official api.py/example.py remain separate from this experiment.
"""

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import threading
import time

from fastapi import FastAPI
import numpy as np
import uvicorn

from detector_adapter import response_from_ultralytics, validate_class_names
from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from utils import decode_view, validate_response


INFERENCE_CONFIG = {
    'imgsz': 1280, 'rect': True, 'device': 0, 'half': True,
    'conf': 0.001, 'iou': 0.7, 'max_det': 300, 'nms': True,
    'agnostic_nms': False, 'verbose': False, 'save': False,
}


class TrainedDetector:
    """Use original-view Results coordinates and the verified global adapter."""

    def __init__(self, checkpoint, *, startup_warmup=True):
        # Configure writable workspace caches before importing Ultralytics.
        # Its import otherwise touches the user's Roaming configuration path.
        cache = Path(__file__).resolve().parent / '.cache'
        (cache / 'matplotlib').mkdir(parents=True, exist_ok=True)
        os.environ['YOLO_CONFIG_DIR'] = str(cache)
        os.environ['MPLCONFIGDIR'] = str(cache / 'matplotlib')
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ['YOLO_VERBOSE'] = 'false'
        import torch
        from ultralytics import YOLO, settings

        torch.set_num_threads(4)
        disabled_integrations = {}
        for key in ('sync', 'clearml', 'comet', 'dvc', 'mlflow', 'raytune', 'tensorboard', 'wandb'):
            if key in settings:
                settings[key] = False
                disabled_integrations[key] = settings[key]
        self.runtime = {
            'torch_num_threads': torch.get_num_threads(),
            'torch_num_interop_threads': torch.get_num_interop_threads(),
            'torch_version': torch.__version__,
            'torch_cuda_build': torch.version.cuda,
            'ultralytics_version': importlib.metadata.version('ultralytics'),
            'environment': {key: os.environ[key] for key in
                            ('YOLO_CONFIG_DIR', 'MPLCONFIGDIR', 'YOLO_AUTOINSTALL', 'YOLO_VERBOSE')},
            'disabled_integrations': disabled_integrations,
        }

        self.checkpoint = Path(checkpoint).resolve(strict=True)
        if self.checkpoint.suffix.lower() != '.pt':
            raise ValueError('A local trained .pt checkpoint is required')
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.model = YOLO(str(self.checkpoint))
        validate_class_names(self.model.names)
        self.lock = threading.Lock()
        self._synchronize = torch.cuda.synchronize
        self.runtime['startup_warmup'] = {
            'enabled': bool(startup_warmup), 'completed': False,
            'input': 'synthetic zero BGR image', 'shape': [540, 960, 3],
            'iterations': 0, 'duration_ms': None,
        }
        if startup_warmup:
            self.warm_up()

    def warm_up(self):
        """Complete CUDA startup before the standalone server can accept requests."""
        with self.lock:
            started = time.perf_counter()
            self.model.predict(source=np.zeros((540, 960, 3), dtype=np.uint8),
                               **INFERENCE_CONFIG)
            self._synchronize()
            self.runtime['startup_warmup'].update(
                completed=True, iterations=1,
                duration_ms=(time.perf_counter() - started) * 1000,
            )

    def __call__(self, request):
        # A timed-out client can leave work in flight. Serialize access rather
        # than allowing concurrent calls to mutate Ultralytics predictor state.
        with self.lock:
            image = decode_view(request.view)
            if image.shape != (request.view.height, request.view.width, 3):
                raise ValueError('Decoded image dimensions do not match the request')
            results = self.model.predict(source=image, **INFERENCE_CONFIG)
            if len(results) != 1:
                raise ValueError('Exactly one prediction result is required')
            # xyxy is already restored to the original 960x540 view. Applying
            # another inverse letterbox transform here would corrupt geometry.
            return response_from_ultralytics(request, results[0])


def create_app(detector):
    app = FastAPI(title='Local fixed-L0 baseline')

    @app.get('/')
    def health():
        return {'service': 'local-trained-detector', 'camera_policy': 'hold'}

    @app.post('/predict', response_model=DroneFlybyPredictResponseDto)
    def predict(request: DroneFlybyPredictRequestDto):
        response = detector(request)
        validate_response(response)
        if response.request_id != request.request_id or response.frame != request.frame:
            raise ValueError('Detector must echo request identity')
        if response.requested_view is not None:
            raise ValueError('This baseline must hold its current view')
        return response

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--port', type=int, default=9054)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    started = time.perf_counter()
    detector = TrainedDetector(args.checkpoint)
    print(f'Initialized detector in {time.perf_counter() - started:.2f}s; '
          f'SHA256={detector.checkpoint_sha256}; '
          f'torch threads={detector.runtime["torch_num_threads"]}; '
          f'startup warm-up={detector.runtime["startup_warmup"]["duration_ms"]:.1f}ms', flush=True)
    # No host override is exposed: this script is an integration experiment.
    uvicorn.run(create_app(detector), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
