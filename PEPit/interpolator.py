import importlib.util
import cvxpy as cp

from PEPit.point import Point


class Interpolator(object):
    """
    A :class:`Interpolator` object encodes an interpolating procedure for a function class or an operator.

    Warnings:
        This class must be overwritten by a child class that encodes one (or more) specific interpolation procedure(s).
        In particular, the method `evaluate` must be overwritten.
        See the :class:`PEPit.interpolators` modules.
        
    Attributes:
        func (Function): A :class:`Function` object that contains the samples to be interpolated.
        d (int): Size of the SDP that it corresponds to (dimension of the space where the interpolation takes place).

    """
    def __init__(self, func):
        self.func = func
        self.d = Point.counter
        self._solver_choice()

    def evaluate(self, x):
        """
        Warnings:
            Needs to be overwritten with an appropriate evaluation/interpolations procedure for the class.

        Raises:
            NotImplementedError: This method must be overwritten in children classes

        """

        raise NotImplementedError("This method must be overwritten in children classes")
        	
    def __call__(self, x):
        return self.evaluate(x)
        
    def _solver_choice(self):
        # If MOSEK is installed, CVXPY will run it.
        # We need to check the presence of a license and handle it in case there is no valid license.
        is_mosek_installed = importlib.util.find_spec("mosek")
        self.solver = cp.MOSEK
        if is_mosek_installed:
            # Import mosek.
            import mosek
            
            # Create an environment.
            mosek_env = mosek.Env()
            
            # Grab the license if there is one.
            try:
                mosek_env.checkoutlicense(mosek.feature.pton)
            except mosek.Error:
                pass
                
            # Check validity of a potentially found license.
            if not mosek_env.expirylicenses() >= 0:
                # In case the license is not valid, ask CVXPY to run SCS.
                self.solver = cp.SCS

        else:
            # If mosek is not installed, ask CVXPY to run SCS.
            self.solver = cp.SCS
