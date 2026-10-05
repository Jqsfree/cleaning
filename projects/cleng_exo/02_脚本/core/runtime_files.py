"""Atomic runtime files and advisory locks, separate from source code."""
from __future__ import annotations
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'-', dir=str(path.parent))
    temp = Path(name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def file_lock(path, *, blocking=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise ValueError('Another operation holds the lock: '+str(path)) from exc
        try: yield
        finally: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def cache_root():
    configured = os.environ.get('EXO_CACHE_DIR')
    return Path(configured).expanduser() if configured else Path(tempfile.gettempdir())/('exo-cache-'+str(os.getuid()))
