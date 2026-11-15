"""Small explicit contracts and immutable, content-addressed local results."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


class ContractError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise ContractError(code)


def number(value, code, low=None, high=None):
    require(type(value) in (int, float) and math.isfinite(value), code)
    require(low is None or value >= low, code)
    require(high is None or value <= high, code)
    return value


def integer(value, code, low=0, high=1000000):
    require(type(value) is int and low <= value <= high, code)
    return value


def token(value, code="TOKEN"):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", value), code)
    return value


def canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ContractError("NON_CANONICAL_JSON") from error


def fields(value, names, code):
    require(isinstance(value, dict) and set(names) <= set(value), code)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def persist(root, payload):
    """Publish a complete JSON once. Existing bytes must agree; never overwrite.

    Atomic hard-link publication on a local filesystem. Not a distributed ledger,
    signature, remote exactly-once guarantee, or power-loss durability claim.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    data = (canonical(payload) + "\n").encode()
    target = root / (digest(payload) + ".json")
    with tempfile.NamedTemporaryFile(dir=root, prefix=".pending-", delete=False) as tmp:
        temporary = Path(tmp.name)
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
    reused = False
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            require(target.is_file() and not target.is_symlink() and target.read_bytes() == data,
                    "ARTIFACT_CONFLICT")
            reused = True
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(target), "sha256": digest(payload), "reused": reused}
