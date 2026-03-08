from PEPit.point import Point

class Interpolator(object):
    """
    A :class:`Interpolator` object encodes an interpolating procedure for a function class or an operator.

    Warnings:
        This class must be overwritten by a child class that encodes one (or more) specific interpolation procedure(s).
        In particular, the method `evaluate` must be overwrten.
        See the :class:`PEPit.interpolators` modules.
        
    Attributes:
        func (Function): A :class:`Function` object that contains the samples to be interpolated.
        d (int): Size of the SDP that it corresponds to (dimension of the space where the interpolation takes place).

    """
    def __init__(self, func):
        self.func = func
        self.d = Point.counter
        # MUST CHECK THAT THE PROBLEM WAS EVALUATED


    def evaluate(self, x):
        """
        Warnings:
            Needs to be overwritten with an appropriate evaluation/interpolations procedure for the class.

        Raises:
            NotImplementedError: This method must be overwritten in children classes

        """

        raise NotImplementedError("This method must be overwritten in children classes")
