# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

from typing import Dict, List, Tuple

import numpy as np


def shape_leq(shape1: Tuple, shape2: Tuple) -> bool:
    assert len(shape1) == len(shape2)
    return all(x1 <= x2 for x1, x2 in zip(shape1, shape2))


def shape_union(*shapes: List[Tuple]) -> Tuple:
    if len(shapes) == 1:
        return shapes[0]
    for s in shapes:
        assert len(s) == len(shapes[0])
    return tuple(max(*dims) for dims in zip(*shapes))


def split_to_shapes(x: np.ndarray, shapes: Dict, axis: int = -1):
    """Split an array and reshape to desired shapes.

    Args:
        x (np.ndarray): The array to be splitted
        shapes (Dict): Dict of shapes (tuples) specifying how to split.
        axis (int): Split dimension.

    Returns:
        List: Splitted observations.
    """
    axis = len(x.shape) + axis if axis < 0 else axis
    split_lengths = [np.prod(shape) for shape in shapes.values()]
    assert x.shape[axis] == sum(split_lengths)
    accum_split_lengths = [sum(split_lengths[:i]) for i in range(1, len(split_lengths))]
    splitted_x = np.split(x, accum_split_lengths, axis)
    return {
        k: x.reshape(*x.shape[:axis], *shape, *x.shape[axis + 1 :])
        for x, (k, shape) in zip(splitted_x, shapes.items())
    }
