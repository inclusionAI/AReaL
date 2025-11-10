"""Tests for dynamic import utilities."""

import pytest

from areal.utils.dynamic_import import (
    import_callable_from_string,
    import_class_from_string,
)


class TestImportClassFromString:
    def test_valid_class_import(self):
        """Test importing a valid class."""
        WorkflowClass = import_class_from_string(
            "areal.api.workflow_api.RolloutWorkflow"
        )
        assert WorkflowClass.__name__ == "RolloutWorkflow"

    def test_invalid_module_path(self):
        """Test error when module doesn't exist."""
        with pytest.raises(ImportError, match="Failed to import module"):
            import_class_from_string("nonexistent.module.SomeClass")

    def test_invalid_class_name(self):
        """Test error when class doesn't exist in module."""
        with pytest.raises(AttributeError, match="has no class"):
            import_class_from_string("areal.api.workflow_api.NonExistentClass")

    def test_invalid_format_no_dot(self):
        """Test error with invalid format (no dots)."""
        with pytest.raises(ValueError, match="Invalid module path"):
            import_class_from_string("NoDotsHere")

    def test_not_a_class(self):
        """Test error when imported object is not a class."""
        with pytest.raises(TypeError, match="is not a class"):
            import_class_from_string(
                "areal.utils.dynamic_import.import_class_from_string"
            )


class TestImportCallableFromString:
    def test_valid_function_import(self):
        """Test importing a valid function."""
        func = import_callable_from_string("areal.utils.data.concat_padded_tensors")
        assert callable(func)
        assert func.__name__ == "concat_padded_tensors"

    def test_invalid_module_path(self):
        """Test error when module doesn't exist."""
        with pytest.raises(ImportError, match="Failed to import module"):
            import_callable_from_string("nonexistent.module.some_func")

    def test_invalid_callable_name(self):
        """Test error when callable doesn't exist in module."""
        with pytest.raises(AttributeError, match="has no attribute"):
            import_callable_from_string("areal.utils.data.nonexistent_function")

    def test_invalid_format(self):
        """Test error with invalid format."""
        with pytest.raises(ValueError, match="Invalid module path"):
            import_callable_from_string("NoDotsHere")

    def test_not_callable(self):
        """Test error when imported object is not callable."""
        # Import a module that's not callable
        with pytest.raises(TypeError, match="is not callable"):
            import_callable_from_string("areal.utils.logging.logger")
