from .alternate_projections import wc_alternate_projections
from .averaged_projections import wc_averaged_projections
from .dykstra import wc_dykstra
from .frank_wolfe import wc_frank_wolfe
from .gradient_descent import wc_gradient_descent
from .halpern_iteration import wc_halpern_iteration
from .inexact_gradient import wc_inexact_gradient
from .optimized_gradient import wc_optimized_gradient
from .proximal_point import wc_proximal_point

__all__ = ['alternate_projections', 'wc_alternate_projections',
           'averaged_projections', 'wc_averaged_projections',
           'dykstra', 'wc_dykstra',
           'frank_wolfe', 'wc_frank_wolfe',
           'gradient_descent', 'wc_gradient_descent',
           'halpern_iteration', 'wc_halpern_iteration',
           'inexact_gradient', 'wc_inexact_gradient',
           'optimized_gradient', 'wc_optimized_gradient',
           'proximal_point', 'wc_proximal_point',
           ]
