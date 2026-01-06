"""Chess logic package initialization"""
from .move_encoding import CanonicalMoveEncoder
from .board_encoding import BoardEncoder

__all__ = ['CanonicalMoveEncoder', 'BoardEncoder']
