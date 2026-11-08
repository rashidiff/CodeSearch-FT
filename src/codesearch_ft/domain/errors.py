"""Typed application errors with concise, user-actionable messages."""


class CodeSearchError(Exception):
    """Base class for expected errors safe to report to a local user."""


class ConfigurationError(CodeSearchError):
    """Raised when a configuration file cannot be read or validated."""


class ModelUnavailableError(CodeSearchError):
    """Raised when a requested model or its optional runtime is unavailable."""


class IndexCompatibilityError(CodeSearchError):
    """Raised when a persisted vector index is incompatible with its encoder."""


class IndexNotFoundError(CodeSearchError):
    """Raised when a requested semantic index ID does not exist."""


class StorageError(CodeSearchError):
    """Raised when local metadata storage cannot be used safely."""


class IndexingError(CodeSearchError):
    """Raised when a repository cannot be indexed or reconciled safely."""
