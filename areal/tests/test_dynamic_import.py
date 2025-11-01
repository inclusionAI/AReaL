import pytest

from areal.utils.dynamic_import import import_from_string


class TestImportFromString:
    def test_import_class(self):
        WorkflowClass = import_from_string("areal.api.workflow_api.RolloutWorkflow")
        assert WorkflowClass.__name__ == "RolloutWorkflow"
        assert isinstance(WorkflowClass, type)

    def test_import_function(self):
        func = import_from_string("areal.utils.data.concat_padded_tensors")
        assert callable(func)
        assert func.__name__ == "concat_padded_tensors"

    def test_invalid_module(self):
        with pytest.raises(ImportError, match="Failed to import module"):
            import_from_string("nonexistent.module.SomeClass")

    def test_invalid_attribute(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            import_from_string("areal.utils.data.nonexistent_function")

    def test_invalid_format_no_dot(self):
        with pytest.raises(ValueError, match="Invalid module path"):
            import_from_string("NoDotsHere")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Invalid module path"):
            import_from_string("")

    def test_none_input(self):
        with pytest.raises(ValueError, match="Invalid module path"):
            import_from_string(None)
