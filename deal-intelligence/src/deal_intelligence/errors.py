"""Shared application errors."""

from __future__ import annotations


class PipelineError(Exception):
    """A user-facing application failure with an HTTP status and stable error code."""

    def __init__(self, code: str, message: str, http_status: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
