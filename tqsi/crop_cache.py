"""Optional lossless disk cache of resized crops, before random augmentation."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def cached_crop(reader, path, box, size, rgb=False, directory=None):
    if not directory:
        return reader(path, box, size, rgb=rgb)
    path = Path(path)
    stat = path.stat()
    # Version the key when resizing/conversion semantics change.
    key = hashlib.sha256(json.dumps([
        1, str(path.resolve()), stat.st_size, stat.st_mtime_ns,
        list(box), size, rgb,
    ]).encode()).hexdigest()
    folder = Path(directory) / key[:2]
    destination = folder / f"{key}.npy"
    try:
        return np.load(destination, allow_pickle=False)
    except FileNotFoundError:
        pass
    value = reader(path, box, size, rgb=rgb)
    folder.mkdir(parents=True, exist_ok=True)
    # Workers/ranks may request the same crop concurrently. Publish atomically.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, value, allow_pickle=False)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return value
