"""A NumPy transformer encoder layer with explicit backpropagation.

The package intentionally contains no autograd engine and no tensor class:
``TransformerEncoderLayer`` operates on plain ``numpy.ndarray`` objects and
implements ``forward`` / ``backward`` by hand.
"""

from . import init, nn
from .nn import TransformerEncoderLayer

__all__ = ["TransformerEncoderLayer", "init", "nn"]
