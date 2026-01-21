import pytest
from sympy import Symbol, sin

from .context import assert_equal

x = Symbol("x", real=True)


def test_left_right_cdot():
    assert_equal("\\sin\\left(x\\right)\\cdot x", sin(x) * x)
