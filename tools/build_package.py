"""Build a hash-verified local deployment package; no download or server startup."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from baseline_endpoint import INFERENCE_CONFIG
from dtos import OBJECT_CLASSES
from camera_policy import MemoryCameraPolicy

FINAL_SHA256 = '4c13fdfd807fc843257fa58d6bf319cf216c1f96042e570be841ccd43e8493bf'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/participant')
    parser.add_argument('--expected-sha256', default=FINAL_SHA256,
                        help='Explicitly override when using a different compatible checkpoint.')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    if sha(checkpoint) != args.expected_sha256:
        parser.error('Checkpoint SHA256 mismatch; original final L0 E50 is the default.')
    versions = {n: importlib.metadata.version(n) for n in
                ['torch', 'torchvision', 'ultralytics', 'numpy', 'opencv-python',
                 'fastapi', 'uvicorn', 'pydantic', 'scipy']}
    output = args.output.resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        parser.error('Output must not contain the source repository.')
    output.mkdir(parents=True, exist_ok=False)
    for source in (ROOT / 'runtime').glob('*.py'):
        shutil.copy2(source, output / source.name)
    shutil.copy2(checkpoint, output / 'model.pt')
    manifest = {
        'schema_version': 1, 'model_name': 'YOLO L0 E50 with final camera policy',
        'checkpoint': 'model.pt', 'class_names': list(OBJECT_CLASSES),
        'inference': INFERENCE_CONFIG, 'runtime_versions': versions,
        'camera_policy': MemoryCameraPolicy().configuration(),
        'files': {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output / 'b0_package.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    from serve_b0 import verify_package
    verify_package(output / 'b0_package.json')
    print(f'Verified package: {output}')

if __name__ == '__main__':
    main()
