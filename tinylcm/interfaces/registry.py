"""Registry pattern implementation for modular components."""

from typing import Any, Callable, Dict, List, Type, TypeVar, Generic

T = TypeVar('T')


class Registry(Generic[T]):
    """Registry for plugin components."""
    
    def __init__(self, base_type: Type[T]):
        """
        Initialize the registry.
        
        Args:
            base_type: Base type for registered components
        """
        self._base_type = base_type
        self._registry: Dict[str, Type[T]] = {}
    
    def register(self, name: str, cls: Type[T]) -> None:
        """
        Register a component.
        
        Args:
            name: Name for the component
            cls: Class to register
            
        Raises:
            TypeError: If cls is not a subtype of base_type
        """
        if not issubclass(cls, self._base_type):
            raise TypeError(f"{cls.__name__} is not a subtype of {self._base_type.__name__}")
        self._registry[name] = cls
    
    def get(self, name: str) -> Type[T]:
        """
        Get a registered component.
        
        Args:
            name: Name of the component
            
        Returns:
            Registered component
            
        Raises:
            KeyError: If no component with this name is registered
        """
        if name not in self._registry:
            raise KeyError(f"No component registered with name '{name}'")
        return self._registry[name]
    
    def create(self, name: str, *args, **kwargs) -> T:
        """
        Create an instance of a registered component.
        
        Args:
            name: Name of the component
            *args: Positional arguments for the constructor
            **kwargs: Keyword arguments for the constructor
            
        Returns:
            Instance of the component
        """
        cls = self.get(name)
        return cls(*args, **kwargs)
    
    def list_registered(self) -> List[str]:
        """
        Get a list of all registered components.
        
        Returns:
            List of component names
        """
        return list(self._registry.keys())