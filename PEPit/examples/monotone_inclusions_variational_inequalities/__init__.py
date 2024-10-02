from .accelerated_proximal_point import wc_accelerated_proximal_point
from .douglas_rachford_splitting import wc_douglas_rachford_splitting
from .optimal_strongly_monotone_proximal_point import wc_optimal_strongly_monotone_proximal_point
from .optimistic_gradient import wc_optimistic_gradient
from .past_extragradient import wc_past_extragradient
from .proximal_point import wc_proximal_point
from .three_operator_splitting import wc_three_operator_splitting
from .frugal_resolvent_splitting import wc_frugal_resolvent_splitting
from .reduced_frugal_resolvent_splitting import wc_reduced_frugal_resolvent_splitting

__all__ = ['accelerated_proximal_point', 'wc_accelerated_proximal_point',
           'douglas_rachford_splitting', 'wc_douglas_rachford_splitting',
           'optimal_strongly_monotone_proximal_point', 'wc_optimal_strongly_monotone_proximal_point',
           'optimistic_gradient', 'wc_optimistic_gradient',
           'past_extragradient', 'wc_past_extragradient',
           'proximal_point', 'wc_proximal_point',
           'three_operator_splitting', 'wc_three_operator_splitting',
           'frugal_resolvent_splitting', 'wc_frugal_resolvent_splitting',
           'reduced_frugal_resolvent_splitting', 'wc_reduced_frugal_resolvent_splitting'
           ]
