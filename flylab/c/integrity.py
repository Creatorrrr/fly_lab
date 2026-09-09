"""Portable, non-executable data formats and content identities."""
import hashlib
import json
import math
import os
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def reject(x):
        raise ValueError('Non-finite JSON: ' + x)
    with open(path, encoding='utf-8') as f:
        return json.load(f, parse_constant=reject)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as f:
        f.write(canonical(value))
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def bounded_int(value, name, low=0, high=2**53 - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name}: integer {low}..{high} required')
    return value


def finite(value, name, low=-1e9, high=1e9):
    if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name}: finite number {low}..{high} required')
    return float(value)


def boolean(value, name):
    if type(value) is not bool:
        raise ValueError(name + ': boolean required')
    return value


def checked_name(value):
    import re
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', value) or '..' in value:
        raise ValueError('A plain artifact name is required')
    return value
