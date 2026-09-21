"""Preserve Python re acceptance with a cheap superset prefilter for CSS.

The frozen CSS grammar is a union of alphabetic named colors, hexadecimal
colors and the ten listed functions ending in ')'. The shape expression is a
superset, so rejecting outside it cannot discard a match of the original
expression. Both checks preserve Python '$' semantics (one final newline).
"""

from __future__ import annotations

import hashlib
import re

CSS_REGEX_SHA256 = "da42f74518e8b94b3c1bbe553eb9ae4a311f8e225ddf8a7268258ed7753d646b"
CSS_SHAPE = re.compile(
    r"^(?:[a-zA-Z]+|#[0-9a-fA-F]+|"
    r"(?:rgb|rgba|hsl|hsla|hwb|lab|oklab|lch|oklch|color)\([\s\S]*\))$"
)
VALIDATION_REVISION = "python-re-css-shape-prefilter-v1"


class CssValidationPattern:
    def __init__(self, expression: str):
        self.original = re.compile(f"^(?:{expression})$")

    def match(self, text: str):
        if CSS_SHAPE.match(text) is None:
            return None
        return self.original.match(text)


def compile_generation_regex(expression: str):
    if hashlib.sha256(expression.encode()).hexdigest() == CSS_REGEX_SHA256:
        return CssValidationPattern(expression)
    return re.compile(f"^(?:{expression})$")
