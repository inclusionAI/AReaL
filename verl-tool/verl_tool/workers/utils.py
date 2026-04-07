import sys
import inspect
import textwrap
import regex as re
from abc import ABCMeta

class SiblingMarker:
    """
    A marker class to indicate that a class is a sibling class.
    This is used to differentiate sibling classes from other classes in the inheritance hierarchy.
    """
    pass

class SiblingMetaClass(ABCMeta):
    """
        Since ray actor classes cannot be inherited. For better development experience,
        we use a metaclass to handle the inheritance of methods from the parent class and sibling class
        when the sibling class is used as a base class in the actor class.
        
        It simply copies the methods from the sibling class to the new class, inheriting from the same parent class as the sibling class (can be a ray actor class).
        
        Example:
        ```python
        from verl_tool.workers.utils import SiblingMetaClass, SiblingMarker
        parent_class = ...
        sibling_class = ...
        class SiblingClass(parent_class, sibling_class, SiblingMarker, metaclass=SiblingMetaClass):
            def __init__(self, *args, **kwargs):
                # super().__init__(*args, **kwargs) do not call as it's already handled by the metaclass
                # Your custom initialization code here
                # e.g., self.sibling_methods_record will contain the methods from sibling_class
                ...
        ```
    """
    def __new__(mcs, name, bases, attrs):
        # print(f"Creating class {name} with bases {bases} and attrs {attrs}")
        if bases[-1].__name__.endswith('SiblingMarker'):
            bases = bases[:-1]  # Remove the SiblingMarker from bases
            assert len(bases) >= 2, f"SiblingMetaClass requires at least two bases, where the last two are the parent class and sibling class. bases: {bases}"
            parent_class = bases[-2]
            sibling_class = bases[-1]
        else:
            parent_class = None
            sibling_class = None
        if sibling_class and sibling_class in bases:
            # Create a dictionary to store super methods
            sibling_methods_record = {}

            # First pass: get methods defined in new class
            new_methods = {method_name for method_name, method in attrs.items()
                            if callable(method) and not method_name.startswith('__')}
                    
            # Check which methods also exist in sibling_class
            for method_name in new_methods:
                if hasattr(sibling_class, method_name):
                    sibling_methods_record[method_name] = sibling_class.__dict__.get(method_name)
            
            # Store the dictionary in the class
            attrs['sibling_methods_record'] = sibling_methods_record
            
            
            new_init = attrs.get('__init__')
            
            # Get the source code of sibling_class.__init__
            init_source = inspect.getsource(sibling_class.__init__)
            
            # Remove the super().__init__() call using regex
            # This pattern matches "super().__init__()" with optional arguments and whitespace
            modified_source = re.sub(r'super\(\)\.__init__\(.*?\)', '', init_source)
            
            # Create the combined init function
            def combined_init(self, *args, **kwargs):
                # First call parent_class.__init__ if it exists
                if 'super(' in modified_source:
                    parent_class.__init__(self)
                
                # Create a local namespace for execution
                local_vars = {}
                # inspect silbing_class.__init__() to get the arguments
                sig = inspect.signature(sibling_class.__init__)
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()  # Apply any defaults if needed
                local_vars = dict(bound.arguments)
                
                # Execute the modified init body (skipping the def line and indentation)
                # This executes all the code from sibling_class.__init__ except super().__init__()
                module = sys.modules[sibling_class.__module__]
                exec(textwrap.dedent(modified_source.split('\n', 1)[1]), module.__dict__, local_vars)

                # Call the new_init if it exists
                if new_init:
                    new_init(self, *args, **kwargs)

            attrs['__init__'] = combined_init
            
            # Copy other methods
            for method_name, method in sibling_class.__dict__.items():
                if not method_name.startswith('__') and method_name not in attrs:
                    attrs[method_name] = method
            
            # Fix bases to avoid duplication
            new_bases = []
            for base in bases:
                if base is sibling_class:
                    if parent_class not in new_bases:
                        new_bases.append(parent_class)
                elif base not in new_bases:
                    new_bases.append(base)
            
            bases = tuple(new_bases)
        
        return super().__new__(mcs, name, bases, attrs)