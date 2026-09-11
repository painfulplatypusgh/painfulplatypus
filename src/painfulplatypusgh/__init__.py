from painfulplatypusgh._core import hello_from_bin
from .differential import diff
from . import matrix

__all__ = ['diff', 'hello', 'matrix']
def hello() -> str:
    return hello_from_bin()
