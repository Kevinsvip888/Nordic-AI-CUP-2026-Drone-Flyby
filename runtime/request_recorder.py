"""Bounded background recording; failures must never invalidate predictions."""

import base64
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import queue
import threading
import time

LOGGER = logging.getLogger('b0.recorder')


def write_json_atomic(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=True, allow_nan=False, indent=2), encoding='utf-8')
    # Windows readers can briefly prevent replacing an open status file.
    # Retry only this sharing failure, on the writer thread, with a hard bound.
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.01 * 2 ** attempt, .1))


class RequestRecorder:
    def __init__(self, directory, metadata, *, capacity=16, max_image_chars=8 * 1024 * 1024,
                 max_total_bytes=2 * 1024 ** 3):
        if min(capacity, max_image_chars, max_total_bytes) < 1:
            raise ValueError('Recording limits must be positive')
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.capacity, self.max_image_chars, self.max_total_bytes = capacity, max_image_chars, max_total_bytes
        self._queue = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._dirty = threading.Event()
        self._counts = dict(attempted=0, enqueued=0, written=0, dropped_full=0,
                            dropped_oversize=0, dropped_closed=0, dropped_budget=0,
                            write_errors=0, status_write_errors=0, submission_errors=0,
                            bytes_written=0, bytes_reserved=0)
        self._last_error = None
        write_json_atomic(self.directory / 'run.json', {
            'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'metadata': metadata,
            'limits': {'queued_records': capacity, 'max_image_characters': max_image_chars,
                       'max_total_record_bytes': max_total_bytes},
            'scope': 'Only received observations and returned predictions; not full-resolution source frames or ground truth.',
        })
        self._thread = threading.Thread(target=self._worker, name='b0-recorder', daemon=True)
        self._dirty.set()
        self._thread.start()

    def snapshot(self):
        with self._lock:
            values = dict(self._counts)
            values.update(enabled=True, pending=values['enqueued'] - values['written'] -
                          values['write_errors'] - values['dropped_budget'],
                          closing=self._closing.is_set(), last_error=self._last_error)
            values['complete_so_far'] = values['attempted'] == values['written'] and not any(
                values[key] for key in ('write_errors', 'status_write_errors', 'submission_errors', 'dropped_full',
                                       'dropped_oversize', 'dropped_closed', 'dropped_budget'))
            return values

    def note_submission_error(self, error_type):
        with self._lock:
            self._counts['submission_errors'] += 1
            self._last_error = error_type
            self._dirty.set()

    def submit(self, request, response, *, received_utc, pre_recording_handler_ms, status, error_type):
        # No PNG decoding, hashing, JSON encoding or disk I/O on this path.
        image = request.view.image
        with self._lock:
            self._counts['attempted'] += 1
            index = self._counts['attempted']
            self._dirty.set()
            if self._closing.is_set():
                self._counts['dropped_closed'] += 1
                return False
            if len(image) > self.max_image_chars:
                self._counts['dropped_oversize'] += 1
                return False
            job = {'arrival_id': index, 'received_utc': received_utc,
                   'request': request.model_dump(mode='json', exclude={'view': {'image'}}),
                   'response': response.model_dump(mode='json'), 'image_base64': image,
                   'result': {'status': status, 'error_type': error_type,
                              'pre_recording_handler_ms': pre_recording_handler_ms}}
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                self._counts['dropped_full'] += 1
                return False
            self._counts['enqueued'] += 1
            return True

    def _write_record(self, job):
        image = job.pop('image_base64')
        prefix = f'{job["arrival_id"]:08d}'
        try:
            image_bytes = base64.b64decode(image, validate=True)
            encoding = 'decoded_base64'
            suffix = '.png' if image_bytes.startswith(b'\x89PNG\r\n\x1a\n') else '.bin'
        except (ValueError, TypeError):
            image_bytes, encoding, suffix = image.encode('utf-8'), 'original_base64_text', '.b64.txt'
        image_name = prefix + suffix
        job['image'] = {'file': image_name, 'encoding': encoding,
                        'sha256': hashlib.sha256(image_bytes).hexdigest(), 'bytes': len(image_bytes)}
        serialized = json.dumps(job, ensure_ascii=True, allow_nan=False, indent=2).encode('utf-8')
        record_bytes = len(image_bytes) + len(serialized)
        with self._lock:
            if self._counts['bytes_reserved'] + record_bytes > self.max_total_bytes:
                self._counts['dropped_budget'] += 1
                return
            # Retain the reservation even after a partial write fails, because
            # an orphan file may still occupy disk space.
            self._counts['bytes_reserved'] += record_bytes
        image_path = self.directory / image_name
        temporary = image_path.with_suffix(image_path.suffix + '.tmp')
        temporary.write_bytes(image_bytes)
        temporary.replace(image_path)
        # The JSON file is the commit marker. A crash can leave an orphan image;
        # readers must not count images without a matching committed record.
        record = self.directory / (prefix + '.json')
        temporary = record.with_suffix('.json.tmp')
        temporary.write_bytes(serialized)
        temporary.replace(record)
        with self._lock:
            self._counts['written'] += 1
            self._counts['bytes_written'] += record_bytes

    def _save_status(self):
        if not self._dirty.is_set():
            return
        self._dirty.clear()
        try:
            write_json_atomic(self.directory / 'status.json', self.snapshot())
        except Exception as exc:
            with self._lock:
                self._counts['status_write_errors'] += 1
                self._last_error = type(exc).__name__
            LOGGER.error('Recording status write failed: %s', type(exc).__name__)

    def _worker(self):
        while not self._closing.is_set() or not self._queue.empty():
            try:
                job = self._queue.get(timeout=.1)
            except queue.Empty:
                self._save_status()
                continue
            try:
                self._write_record(job)
            except Exception as exc:
                with self._lock:
                    self._counts['write_errors'] += 1
                    self._last_error = type(exc).__name__
                LOGGER.error('Recording write failed: %s', type(exc).__name__)
            finally:
                self._queue.task_done()
                self._dirty.set()
                self._save_status()
        self._dirty.set()
        self._save_status()

    def flush(self, timeout=10):
        deadline = time.monotonic() + timeout
        while self.snapshot()['pending'] and time.monotonic() < deadline:
            time.sleep(.01)
        return self.snapshot()['pending'] == 0

    def close(self, timeout=10):
        self._closing.set()
        self._dirty.set()
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()
