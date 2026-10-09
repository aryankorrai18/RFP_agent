"""Uploaded files are kept on disk under data/uploads, named by content hash."""

from __future__ import annotations

import hashlib
from pathlib import PurePath
from typing import TYPE_CHECKING

from ...errors import PipelineError
from ...parsing.parser import SUPPORTED_EXTENSIONS
from .db import Document

if TYPE_CHECKING:
    from .context import V1Context


def check_extension(filename: str) -> str:
    suffix = PurePath(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise PipelineError("unsupported_file_type", f"'{suffix or filename}' is not supported. Use one of: {supported}.", 415)
    return suffix


def store_document(ctx: V1Context, kind: str, filename: str, data: bytes) -> Document:
    suffix = check_extension(filename)
    digest = hashlib.sha256(data).hexdigest()
    folder = ctx.settings.uploads_dir
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest}{suffix}"
    if not path.exists():
        path.write_bytes(data)
    with ctx.db.session() as session:
        document = Document(kind=kind, filename=filename, sha256=digest, stored_path=str(path))
        session.add(document)
        session.commit()
    return document
