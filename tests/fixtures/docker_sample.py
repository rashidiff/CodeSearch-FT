from collections.abc import Iterable


def normalize_values_preserving_order(items: Iterable[str]) -> list[str]:
    """Remove duplicates while preserving first occurrence order."""
    return list(dict.fromkeys(items))
