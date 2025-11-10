import importlib


def dynamic_import(module_path, function_name):
    """Dynamically import a function from a module."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, function_name)
    except ImportError as e:
        print(f"Failed to import module {module_path}: {e}")
    except AttributeError:
        print(f"Function {function_name} not found in module {module_path}")
