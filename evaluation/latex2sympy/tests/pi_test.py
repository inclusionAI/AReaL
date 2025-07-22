import pytest
from sympy import Symbol, acos, cos, pi

from .context import _Mul, _Pow, assert_equal


def test_pi_frac():
    assert_equal("\\frac{\\pi}{3}", _Mul(pi, _Pow(3, -1)))


def test_pi_nested():
    assert_equal(
        "\\arccos{\\cos{\\frac{\\pi}{3}}}",
        acos(cos(_Mul(pi, _Pow(3, -1)), evaluate=False), evaluate=False),
    )


def test_pi_arccos():
    assert_equal("\\arccos{-1}", pi, symbolically=True)
