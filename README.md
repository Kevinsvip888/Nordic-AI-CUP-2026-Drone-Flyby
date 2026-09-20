# AeroGaze: Active Camera Object Detection

AeroGaze combines YOLO26-L detection, per-object Kalman tracking, camera-motion compensation, and conservative L0/L1 inspection for wide-area aerial scenes. This repository is a compact post-competition extraction of the final Drone Flyby participant.

This repository contains source and CPU regression tests. **Weights, competition images, annotations, credentials, recordings and historical experiment outputs are not included.** No service starts automatically.

Run `python tools/audit_release.py` before publishing. It checks the staged source list for non-English text, secret-like values, local paths, archives, model files and oversized files.

## Final camera policy

- Receive at least **three consecutive L0 observations** before requesting L1. This is a minimum, not a periodic zoom schedule.
- Among safely associated candidates, prioritize a change in the predicted class of the same object; otherwise prioritize absolute confidence change.
- Observe L1 for one frame, then request L0 immediately. Never request L2.
- Predict each tracked object's position with a Kalman filter and update it from reliable current detections. Compensate for estimated camera motion using only received images.
- Confidence >0.3 establishes short memory; confidence >0.5 retains identity. Position output remains subject to visibility, age and uncertainty checks. Off-crop memory is limited to two unseen frames; source-frame exits are removed.
- Following the final experiment configuration, `ta-ta` and `small_launcher` are excluded from policy output. This is an experiment choice, not a competition requirement.

Class changes compare the latest two actual observations, not all changes over a three-frame window. Tracking cannot recover never-observed objects outside the current crop.

## Layout

```text
runtime/       detector, camera policy, tracking, protocol and local server
tools/         portable deployment package builder
tests/         synthetic CPU regression tests; no weights or private data
docs/          results, provenance and third-party notices
.github/       CPU test workflow
```

`runtime/camera_policy.py` implements camera selection; `tracking.py`, `kalman_position.py` and `visible_view_motion.py` implement memory. `live_adapter.py` provides sequence isolation, duplicate handling and causal updates. The frozen service implementation keeps its existing `serve_b0.py` filename.

## Setup and tests

Use Python **3.12**. CPU policy tests need no CUDA or model weights:

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -v
```

Inference was tested with PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128 and Ultralytics 8.4.155 on an NVIDIA GPU. Install the compatible PyTorch CUDA build for your platform, then:

```sh
python -m pip install -r requirements.txt
```

The serving configuration uses CUDA device 0 and FP16; CPU-only inference is not supported by this frozen configuration. Dependencies are pinned to the recorded environment where available, not advertised as universal platform support.

## Build and run locally

Download the final L0 E50 checkpoint from [AeroGaze YOLO26-L on Hugging Face](https://huggingface.co/Invisible-dog/aerogaze-yolo26l) and place it at `weights/model.pt`. The builder checks its expected SHA256 and refuses to overwrite an existing output directory.

```sh
python tools/build_package.py --checkpoint weights/model.pt --output build/participant
python build/participant/serve_b0.py --verify-only
python build/participant/serve_b0.py --host 127.0.0.1 --port 9054
```

The service exposes `GET /ready` and `POST /predict`. Protocol DTOs are in `runtime/dtos.py`. The model must have the exact 16-class order in that file. To deliberately use another compatible checkpoint, pass its digest through `--expected-sha256`; this does not establish that its accuracy matches the final model.

Inference uses 1280, aspect-ratio-preserving preprocessing, FP16, original/horizontal/vertical/both-flip predictions, top-two class alternatives and box fusion. The returned class scores are not calibrated probabilities.

Optional request recording is opt-in:

```sh
python build/participant/serve_b0.py --record-dir recordings/session-001
```

Recording directories contain input images and predictions and must remain private. Public hosting/tunnels and competition submission are deliberately outside this source repository. The local service has no authentication; keep its default loopback binding unless you configure access controls yourself.

This release is the final inference/policy implementation, not a portable reconstruction of every historical training experiment. Training datasets, calibration/composer assets and historical training scripts remain in the private working project.

## Results and limitations

The final policy passed a 248-frame offline replay with 50 synthetic L1 observations and causal checks. It has **no confirmed complete official Validation score**. The historical best confirmed score, **0.4950687646541436**, belonged to a different E40 fixed-L0 baseline; it must not be attributed to this final policy. See [results](docs/RESULTS.md).

## License

The source code in this repository is available under the [MIT License](LICENSE). Dependencies and separately supplied model weights retain their own licenses.
