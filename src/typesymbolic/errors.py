"""Common base for this package's exceptions. Concrete exceptions keep
their original base too, via multiple inheritance."""

from __future__ import annotations


class TypesymbolicError(Exception):
    """Base for every exception this package raises intentionally."""
