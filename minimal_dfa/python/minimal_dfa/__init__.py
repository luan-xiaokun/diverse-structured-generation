"""This package provides an interface to construct minimal DFAs from regexes."""

from .minimal_dfa_rs import MinDivDFA, DiverseGuideDFA

__all__ = ["MinDivDFA", "DiverseGuideDFA"]
__version__ = "0.1.0"
