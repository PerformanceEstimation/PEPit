from .gradient_descent import wc_gradient_descent
from .no_lips_1 import wc_no_lips_1
from .no_lips_2 import wc_no_lips_2
from .gradient_descent_Lojasiewicz import wc_gradient_descent_naiveLojaciewicz
from .gradient_descent_refinedLojasiewicz import wc_gradient_descent_refinedLojaciewicz
from .gradient_descent_expertrefinedLojasiewicz import wc_gradient_descent_expertLojaciewicz

__all__ = ['gradient_descent', 'wc_gradient_descent',
           'no_lips_1', 'wc_no_lips_1',
           'no_lips_2', 'wc_no_lips_2',
           'gradient_descent_Lojasiewicz', 'wc_gradient_descent_naiveLojaciewicz',
           'gradient_descent_refinedLojasiewicz', 'wc_gradient_descent_refinedLojaciewicz',
           'gradient_descent_expertrefinedLojasiewicz', 'wc_gradient_descent_expertLojaciewicz',
           ]
