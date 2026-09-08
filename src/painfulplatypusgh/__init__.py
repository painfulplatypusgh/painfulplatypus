from painfulplatypusgh._core import hello_from_bin
from .differential import diff

__all__ = ['diff', 'hello']
def hello() -> str:
    return hello_from_bin()
