"""Custom exception classes for Veil."""


class VeilError(Exception):
    """Base class for all Veil-specific exceptions."""


class CapacityError(VeilError):
    """Raised when the message does not fit into the cover image."""


class UnsupportedFormatError(VeilError):
    """Raised when the image format is not supported by Veil."""


class ExtractionError(VeilError):
    """Raised when hidden data cannot be extracted from an image."""
