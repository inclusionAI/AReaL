import pytest
from sympy import Symbol, sin

from .context import assert_equal

x = Symbol("x", real=True)


def test_overline():
    assert_equal(
        "\\frac{\\sin(x)}{\\overline{x}_n}", sin(x) / Symbol("xbar_n", real=True)
    )
