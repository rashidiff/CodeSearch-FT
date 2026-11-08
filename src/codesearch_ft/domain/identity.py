"""Stable digest helpers for semantic index compatibility checks."""

import hashlib
import json
from collections.abc import Iterable


def corpus_fingerprint(functions: Iterable[tuple[int, str]]) -> str:
    """Hash ordered `(function_id, content_sha256)` identities canonically."""
    payload = json.dumps(list(functions), separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
