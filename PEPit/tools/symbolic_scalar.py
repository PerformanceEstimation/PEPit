from fractions import Fraction
import numbers

import sympy as sp


def _to_sympy_number(value, strict=False):
    if isinstance(value, bool):
        return sp.Integer(int(value))
    if isinstance(value, sp.Rational):
        return value
    if isinstance(value, Fraction):
        return sp.Rational(value.numerator, value.denominator)
    if isinstance(value, numbers.Integral):
        return sp.Integer(int(value))
    if strict:
        raise TypeError(
            "SymbolicScalar only accepts exact values: int, fractions.Fraction, or sympy.Rational."
        )
    if isinstance(value, numbers.Real):
        return sp.Rational(str(float(value)))
    raise TypeError("Unsupported scalar type: {}".format(type(value)))


class SymbolicScalar:
    """
    Symbolic scalar with deferred numeric evaluation.

    Arithmetic operations only manipulate symbolic expressions. Numeric values are
    computed when calling ``evaluate`` (directly or through ``evaluate_scalar``).
    """

    def __init__(self, value, symbol=None, _expr=None, _defaults=None):
        if _expr is not None:
            self._expr = sp.simplify(_expr)
            self._defaults = dict(_defaults or {})
            return

        if symbol is None:
            self._expr = _to_sympy_number(value, strict=True)
            self._defaults = {}
        else:
            name = str(symbol)
            self._expr = sp.Symbol(name)
            self._defaults = {name: _to_sympy_number(value, strict=True)}

    @property
    def symbol(self):
        return str(sp.simplify(self._expr))

    @staticmethod
    def _is_numeric(value):
        return isinstance(value, (SymbolicScalar, numbers.Real))

    @staticmethod
    def _merge_defaults(defaults_1, defaults_2):
        merged = dict(defaults_1)
        for key, value in defaults_2.items():
            if key in merged and sp.simplify(merged[key] - value) != 0:
                raise ValueError("Conflicting default value for symbol '{}'.".format(key))
            merged[key] = value
        return merged

    @classmethod
    def _coerce(cls, value):
        if isinstance(value, SymbolicScalar):
            return value._expr, dict(value._defaults)
        return _to_sympy_number(value, strict=False), {}

    @classmethod
    def _from_op(cls, expr, defaults):
        return cls(0, _expr=expr, _defaults=defaults)

    @staticmethod
    def _format_substitutions(substitutions):
        if substitutions is None:
            return {}
        formatted = {}
        for key, value in substitutions.items():
            name = str(key)
            formatted[name] = _to_sympy_number(value, strict=False)
        return formatted

    def evaluate(self, substitutions=None):
        subs = dict(self._defaults)
        subs.update(self._format_substitutions(substitutions))

        sympy_subs = {sp.Symbol(name): value for name, value in subs.items()}
        evaluated = sp.simplify(self._expr.subs(sympy_subs))
        if evaluated.free_symbols:
            raise ValueError(
                "Missing substitutions for symbols: {}".format(
                    ", ".join(sorted(str(s) for s in evaluated.free_symbols))
                )
            )
        return float(sp.N(evaluated))

    def __add__(self, other):
        if not self._is_numeric(other):
            return NotImplemented
        other_expr, other_defaults = self._coerce(other)
        defaults = self._merge_defaults(self._defaults, other_defaults)
        return self._from_op(self._expr + other_expr, defaults)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not self._is_numeric(other):
            return NotImplemented
        other_expr, other_defaults = self._coerce(other)
        defaults = self._merge_defaults(self._defaults, other_defaults)
        return self._from_op(self._expr - other_expr, defaults)

    def __rsub__(self, other):
        if not self._is_numeric(other):
            return NotImplemented
        other_expr, other_defaults = self._coerce(other)
        defaults = self._merge_defaults(other_defaults, self._defaults)
        return self._from_op(other_expr - self._expr, defaults)

    def __mul__(self, other):
        if not self._is_numeric(other):
            return NotImplemented
        other_expr, other_defaults = self._coerce(other)
        defaults = self._merge_defaults(self._defaults, other_defaults)
        return self._from_op(self._expr * other_expr, defaults)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not self._is_numeric(other):
            return NotImplemented
        other_expr, other_defaults = self._coerce(other)
        defaults = self._merge_defaults(self._defaults, other_defaults)
        return self._from_op(self._expr / other_expr, defaults)

    def __rtruediv__(self, other):
        if not self._is_numeric(other):
            return NotImplemented
        other_expr, other_defaults = self._coerce(other)
        defaults = self._merge_defaults(other_defaults, self._defaults)
        return self._from_op(other_expr / self._expr, defaults)

    def __neg__(self):
        return self._from_op(-self._expr, self._defaults)

    def __float__(self):
        return self.evaluate()

    def __repr__(self):
        return "SymbolicScalar({})".format(self.symbol)


def is_scalar(value):
    return isinstance(value, (SymbolicScalar, numbers.Real))


def evaluate_scalar(value, substitutions=None):
    if isinstance(value, SymbolicScalar):
        return value.evaluate(substitutions=substitutions)
    return float(value)
