from .cvxpy_wrapper import CvxpyWrapper
from .mosek_wrapper import MosekWrapper

WRAPPERS = {
    "cvxpy": CvxpyWrapper,
    "mosek": MosekWrapper,
}

__all__ = ['cvxpy_wrapper', 'CvxpyWrapper',
           'mosek_wrapper', 'MosekWrapper',
           'WRAPPERS',
           ]
