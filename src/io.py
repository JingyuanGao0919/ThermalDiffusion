                                                           
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

EXCLUDED = Path(os.environ["THERMAL_FORBIDDEN_ROOT"])


def under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def checked(path, *, root=None, excluded=EXCLUDED, must_exist=True) -> Path:
    original = Path(path).absolute()
                                                                                
    excluded = Path(excluded).absolute()
    if under(original, excluded):
        raise PermissionError(f"Excluded path: {original}")
    resolved = original.resolve(strict=must_exist)
    if under(resolved, excluded):
        raise PermissionError(f"Symlink targets excluded directory: {original}")
    if root is not None and not under(resolved, Path(root).resolve(strict=True)):
        raise PermissionError(f"Path escapes declared root: {original}")
    return resolved


def sha256_file(path, **kwargs):
    path = checked(path, **kwargs)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_json(path):
    with checked(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value, *, overwrite=False):
    path = checked(path, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if not overwrite:
                                                                                  
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        return
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    os.replace(tmp, path)
