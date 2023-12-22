from .cvxpy_wrapper import CvxpyWrapper
from .mosek_wrapper import MosekWrapper

# Define a dictionary of wrapper.
# By convention, the keys must be written with lower cases.
WRAPPERS = {
    "cvxpy": CvxpyWrapper,
    "mosek": MosekWrapper,
}

__all__ = ['cvxpy_wrapper', 'CvxpyWrapper',
           'mosek_wrapper', 'MosekWrapper',
           'WRAPPERS',
           ]
