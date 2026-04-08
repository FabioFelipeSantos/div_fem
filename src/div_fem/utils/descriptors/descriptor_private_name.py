from typing import Any, Callable, Generic, TypeVar

_T = TypeVar("_T")


class PrivateName:

    private_name: str

    def __set_name__(self, owner: Any, name: str) -> None:
        self.private_name = "_" + name


class DescriptorBaseClass(PrivateName, Generic[_T]):

    def __init__(self, *, validation: Callable[[Any], None] | None = None) -> None:
        self.validation = validation

    def __get__(self, obj: Any, objtype=None) -> _T:
        value = getattr(obj, self.private_name, None)

        if not value:
            raise AttributeError(
                f"The {str(obj.__class__.__name__).capitalize()} class hasn't have your value set. Provide a correct place to the class so the atribute be set."
            )

        return value

    def __set__(self, obj: Any, value: _T) -> None:
        try:
            self.validation(value) if self.validation else None
            setattr(obj, self.private_name, value)
        except:
            raise
