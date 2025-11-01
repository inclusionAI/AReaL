"""Utilities for dynamic importing of classes and callables from module paths."""

import importlib
from collections.abc import Callable


def import_class_from_string(module_path: str) -> type:
    """Import a class from a string module path.

    Parameters
    ----------
    module_path : str
        Full module path to the class, e.g., "areal.workflow.rlvr.RLVRWorkflow"

    Returns
    -------
    Type
        The imported class

    Raises
    ------
    ValueError
        If the module path is invalid or doesn't contain a class name
    ImportError
        If the module cannot be imported
    AttributeError
        If the class doesn't exist in the module

    Examples
    --------
    >>> WorkflowClass = import_class_from_string("areal.workflow.rlvr.RLVRWorkflow")
    >>> workflow = WorkflowClass(reward_fn=..., gconfig=..., tokenizer=...)
    """
    if not module_path or not isinstance(module_path, str):
        raise ValueError(
            f"Invalid module path: {module_path!r}. "
            f"Expected a non-empty string like 'module.path.ClassName'."
        )

    parts = module_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid module path: {module_path!r}. "
            f"Expected format 'module.path.ClassName', got {len(parts)} part(s)."
        )

    module_name, class_name = parts

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{module_name}' from path '{module_path}': {e}"
        ) from e

    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' has no class '{class_name}'. "
            f"Available attributes: {dir(module)}"
        ) from e

    if not isinstance(cls, type):
        raise TypeError(
            f"'{module_path}' resolved to {cls!r}, which is not a class. "
            f"Expected a class type."
        )

    return cls


def import_callable_from_string(module_path: str) -> Callable:
    """Import a callable (function or class) from a string module path.

    Parameters
    ----------
    module_path : str
        Full module path to the callable, e.g., "my_module.my_filter_function"

    Returns
    -------
    Callable
        The imported callable

    Raises
    ------
    ValueError
        If the module path is invalid or doesn't contain a callable name
    ImportError
        If the module cannot be imported
    AttributeError
        If the callable doesn't exist in the module
    TypeError
        If the imported object is not callable

    Examples
    --------
    >>> filter_fn = import_callable_from_string("my_module.my_filter")
    >>> result = filter_fn(data)
    """
    if not module_path or not isinstance(module_path, str):
        raise ValueError(
            f"Invalid module path: {module_path!r}. "
            f"Expected a non-empty string like 'module.path.function_name'."
        )

    parts = module_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid module path: {module_path!r}. "
            f"Expected format 'module.path.function_name', got {len(parts)} part(s)."
        )

    module_name, callable_name = parts

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{module_name}' from path '{module_path}': {e}"
        ) from e

    try:
        func = getattr(module, callable_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' has no attribute '{callable_name}'. "
            f"Available attributes: {dir(module)}"
        ) from e

    if not callable(func):
        raise TypeError(
            f"'{module_path}' resolved to {func!r}, which is not callable. "
            f"Expected a function or callable object."
        )

    return func
