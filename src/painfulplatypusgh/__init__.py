from painfulplatypusgh._core import hello_from_bin
from .differential import diff
from . import matrix
from . import distributions
from . import model

__all__ = ['diff', 'hello', 'matrix', 'distributions', 'model']
def hello() -> str:
    return hello_from_bin()
