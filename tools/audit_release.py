"""Audit staged source files for public release safety."""

from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {'.env', '.key', '.pem', '.pt', '.pth', '.onnx', '.engine',
                      '.zip', '.png', '.jpg', '.jpeg', '.log'}
SECRET_PATTERNS = {
    'private key': rb'-----BEGIN [A-Z ]*PRIVATE KEY-----',
    'GitHub token': rb'(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}',
    'generic secret assignment': rb'(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*["\'][^"\']{8,}',
    'temporary tunnel': rb'https://[^\s"\']+\.trycloudflare\.com',
    'Windows user path': rb'(?i)[A-Z]:\\Users\\',
    'Windows workspace path': rb'(?i)[A-Z]:\\Github[^\r\n"\']*',
}


def staged_files() -> list[Path]:
    raw = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT)
    return [ROOT / item.decode('utf8') for item in raw.split(b'\0') if item]


def main() -> None:
    files = staged_files()
    if not files:
        raise SystemExit('No staged source files found')
    failures = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file():
            failures.append(f'{relative}: not a regular file')
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name == '.env':
            failures.append(f'{relative}: forbidden release file type')
        data = path.read_bytes()
        if len(data) > 1_000_000:
            failures.append(f'{relative}: file exceeds 1 MB')
        try:
            data.decode('ascii')
        except UnicodeDecodeError:
            failures.append(f'{relative}: contains non-ASCII text')
        for label, pattern in SECRET_PATTERNS.items():
            if re.search(pattern, data):
                failures.append(f'{relative}: matched {label}')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'PASS: {len(files)} staged files contain ASCII source only and no known private-data patterns')


if __name__ == '__main__':
    main()
