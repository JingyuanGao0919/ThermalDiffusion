                                                                             
from __future__ import annotations

import math
import struct
from pathlib import Path
from .io import checked

                                                                             
PARAM_COUNTS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12,
                7: 5, 8: 4, 9: 5, 10: 12}


def unpack(stream, fmt):
    n = struct.calcsize("<" + fmt)
    raw = stream.read(n)
    if len(raw) != n:
        raise ValueError("Truncated COLMAP binary")
    return struct.unpack("<" + fmt, raw)


def read_cameras(path):
    result = {}
    with checked(path).open("rb") as stream:
        count, = unpack(stream, "Q")
        if count > 100000:
            raise ValueError("Implausible camera count")
        for _ in range(count):
            uid, model, width, height = unpack(stream, "iiQQ")
            if model not in PARAM_COUNTS:
                raise ValueError(f"Unsupported COLMAP camera model {model}")
            params = unpack(stream, "d" * PARAM_COUNTS[model])
            if uid in result:
                raise ValueError(f"Duplicate camera ID {uid}")
            if model not in (0, 1):
                                                                                 
                raise ValueError(f"Undistort camera model {model} before reproduction")
            fx, fy, cx, cy = ((params[0], params[0], params[1], params[2])
                              if model == 0 else params)
            if min(width, height, fx, fy) <= 0 or not all(map(math.isfinite, params)):
                raise ValueError("Invalid camera intrinsics")
            result[uid] = dict(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy,
                               model="PINHOLE" if model == 1 else "SIMPLE_PINHOLE")
        if stream.read(1):
            raise ValueError("Trailing data in cameras.bin")
    return result


def rotation(q):
    n = math.sqrt(sum(x*x for x in q))
    if n < 1e-12 or not math.isfinite(n):
        raise ValueError("Invalid camera quaternion")
    w, x, y, z = (v/n for v in q)
    return [[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]]


def read_images(path):
    result = {}
    with checked(path).open("rb") as stream:
        size = Path(path).stat().st_size
        count, = unpack(stream, "Q")
        if count > 1000000:
            raise ValueError("Implausible image count")
        for _ in range(count):
            uid, = unpack(stream, "i")
            q = unpack(stream, "dddd")
            t = unpack(stream, "ddd")
            camera_id, = unpack(stream, "i")
            raw = bytearray()
            while True:
                char = stream.read(1)
                if not char:
                    raise ValueError("Unterminated COLMAP image name")
                if char == b"\0":
                    break
                raw.extend(char)
                if len(raw) > 4096:
                    raise ValueError("Implausible image name")
            name = raw.decode("utf-8")
            if Path(name).name != name or "\\" in name or not name:
                raise ValueError(f"Nonlocal image name {name!r}")
            points, = unpack(stream, "Q")
            if stream.tell() + points*24 > size:
                raise ValueError("Truncated COLMAP point observations")
            stream.seek(points*24, 1)
            if name in result or not all(map(math.isfinite, t)):
                raise ValueError("Duplicate name or invalid extrinsics")
            result[name] = dict(colmap_image_id=uid, camera_id=camera_id,
                                R=rotation(q), T=list(t))
        if stream.read(1):
            raise ValueError("Trailing data in images.bin")
    return result
