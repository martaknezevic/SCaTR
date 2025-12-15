"""
Method registry for dynamic method loading.

Provides a centralized registry for all available methods.
"""

from typing import Dict, Type, Any, Optional
from pathlib import Path
import importlib
import importlib.util
import inspect

from methods.base_method import BaseMethod


class MethodRegistry:
    """
    Registry for confidence scoring methods.

    Provides factory pattern for creating method instances.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._methods: Dict[str, Type[BaseMethod]] = {}

    def register(self, name: str, method_class: Type[BaseMethod]):
        """
        Register a method class.

        Args:
            name: Name of the method
            method_class: Method class (must inherit from BaseMethod)
        """
        if not issubclass(method_class, BaseMethod):
            raise ValueError(f"Method class must inherit from BaseMethod: {method_class}")

        self._methods[name] = method_class

    def get(self, name: str) -> Type[BaseMethod]:
        """
        Get method class by name.

        Args:
            name: Name of the method

        Returns:
            Method class
        """
        if name not in self._methods:
            raise ValueError(f"Unknown method: {name}. Available: {list(self._methods.keys())}")

        return self._methods[name]

    def create(self, name: str, config: Dict[str, Any],
               common_params: Optional[Dict[str, Any]] = None,
               **kwargs) -> BaseMethod:
        """
        Create method instance.

        Args:
            name: Name of the method
            config: Method-specific configuration
            common_params: Common parameters (e.g., tail_tokens, device)
            **kwargs: Additional keyword arguments for specific methods

        Returns:
            Method instance
        """
        method_class = self.get(name)

        # Merge configs
        full_config = {}
        if common_params:
            full_config.update(common_params)
        full_config.update(config)

        # Check method signature to determine how to instantiate
        sig = inspect.signature(method_class.__init__)
        params = list(sig.parameters.keys())

        # Different methods have different signatures
        # Some take (name, config), others take (**config), some need extra args

        try:
            # Try **config pattern first (base_transformer, attention_matrix_metric)
            if 'name' not in params and 'config' not in params and len(params) > 1:
                instance = method_class(**full_config, **kwargs)
            # Try (config) pattern (most methods)
            elif 'config' in params:
                instance = method_class(full_config)
            # Try (name, config) pattern (legacy)
            elif 'name' in params and 'config' in params:
                instance = method_class(name, full_config, **kwargs)
            else:
                # Fallback: just pass config
                instance = method_class(full_config, **kwargs)

            return instance

        except Exception as e:
            raise ValueError(f"Failed to create method '{name}': {e}") from e

    def list_methods(self) -> list:
        """List all registered methods."""
        return list(self._methods.keys())

    def auto_discover_methods(self, methods_dir: Path):
        """
        Auto-discover and register methods from directory.

        Args:
            methods_dir: Directory containing method implementations
        """
        import sys

        methods_dir = Path(methods_dir)

        if not methods_dir.exists():
            return

        # Import all Python files in methods directory
        for py_file in methods_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            # Import module
            module_name = py_file.stem
            print(module_name)
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find BaseMethod subclasses
                    found_classes = 0
                    registered_classes = 0
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        print(name, obj)
                        if issubclass(obj, BaseMethod) and obj != BaseMethod:
                            found_classes += 1
                            if obj.__module__ == module.__name__:
                                # Register with snake_case name
                                method_name = self._class_name_to_method_name(name)
                                self.register(method_name, obj)
                                registered_classes += 1
                                print(f"  ✓ Registered method: {method_name} ({name})")

                    if found_classes > 0 and registered_classes == 0:
                        print(f"  ⚠ Found {found_classes} BaseMethod subclasses in {py_file.name} but none matched module name")

            except Exception as e:
                print(f"  ✗ Failed to load method from {py_file}: {e}")
                import traceback
                traceback.print_exc()

    def _class_name_to_method_name(self, class_name: str) -> str:
        """
        Convert class name to method name.

        Examples:
            RandomMethod -> random
            ConvexOptimizationMethod -> convex_optimization
            BaseTransformerMethod -> base_transformer
        """
        # Remove 'Method' suffix
        if class_name.endswith('Method'):
            class_name = class_name[:-6]

        # Convert CamelCase to snake_case
        result = []
        for i, char in enumerate(class_name):
            if char.isupper() and i > 0:
                result.append('_')
            result.append(char.lower())

        return ''.join(result)


# Global registry instance
_global_registry = None


def get_global_registry() -> MethodRegistry:
    """Get or create global method registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = MethodRegistry()
    return _global_registry


def register_method(name: str):
    """
    Decorator to register a method class.

    Usage:
        @register_method('my_method')
        class MyMethod(BaseMethod):
            ...
    """
    def decorator(cls):
        get_global_registry().register(name, cls)
        return cls
    return decorator
