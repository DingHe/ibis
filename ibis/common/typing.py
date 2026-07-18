from __future__ import annotations

import inspect
import re
import sys
from abc import abstractmethod
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Optional, TypeVar, get_args, get_origin
from typing import get_type_hints as _get_type_hints

from ibis.common.bases import Abstract
from ibis.common.caching import memoize

if TYPE_CHECKING:
    from typing_extensions import Self


from types import UnionType
from typing import TypeAlias

# Keep this alias in sync with unittest.case._ClassInfo
_ClassInfo: TypeAlias = type | UnionType | tuple["_ClassInfo", ...]


T = TypeVar("T")
U = TypeVar("U")

Namespace = dict[str, Any]
VarTuple = tuple[T, ...]


@memoize
def get_type_hints(
    obj: Any,
    include_extras: bool = True,
    include_properties: bool = False,
) -> dict[str, Any]:
    """Get type hints for a callable or class.

    Extension of typing.get_type_hints that supports getting type hints for
    class properties.

    Parameters
    ----------
    obj
        Callable or class to get type hints for.
    include_extras
        Whether to include extra type hints such as Annotated.
    include_properties
        Whether to include type hints for class properties.

    Returns
    -------
    Mapping of parameter or attribute name to type hint.

    """
    try:
        hints = _get_type_hints(obj, include_extras=include_extras)
    except TypeError:
        return {}

    if include_properties:
        for name in dir(obj):
            # https://docs.python.org/3/library/functions.html#dir
            # it's entirely possible for attributes to come out of `dir(obj)`
            # that don't exist on the `obj`
            #
            # dir is designed to inform interactive use, not to consistently or
            # rigorously defined
            #
            # https://github.com/great-expectations/great_expectations/issues/9698#issuecomment-2051252373
            # is another in-the-wild example of this
            attr = getattr(obj, name, None)
            if isinstance(attr, property):
                annots = _get_type_hints(attr.fget, include_extras=include_extras)
                if return_annot := annots.get("return"):
                    hints[name] = return_annot

    return hints


@memoize
def get_type_params(obj: Any) -> dict[str, type]:
    """Get type parameters for a generic class.

    Parameters
    ----------
    obj
        Generic class to get type parameters for.

    Returns
    -------
    Mapping of type parameter name to type.

    Examples
    --------
    >>> from typing import Dict, List
    >>> class MyList(List[T]): ...
    >>> get_type_params(MyList[int])
    {'T': <class 'int'>}
    >>> class MyDict(Dict[T, U]): ...
    >>> get_type_params(MyDict[int, str])
    {'T': <class 'int'>, 'U': <class 'str'>}

    """
    args = get_args(obj)
    origin = get_origin(obj) or obj
    bases = getattr(origin, "__orig_bases__", ())
    params = getattr(origin, "__parameters__", ())

    result = {}
    for base in bases:
        result.update(get_type_params(base))

    param_names = (p.__name__ for p in params)
    result.update(zip(param_names, args))

    return result


@memoize
def get_bound_typevars(obj: Any) -> dict[TypeVar, tuple[str, type]]:
    """Get type variables bound to concrete types for a generic class.

    Parameters
    ----------
    obj
        Generic class to get type variables for.

    Returns
    -------
    Mapping of type variable to attribute name and type.

    Examples
    --------
    >>> from typing import Generic
    >>> class MyStruct(Generic[T, U]):
    ...     a: T
    ...     b: U
    >>> get_bound_typevars(MyStruct[int, str])
    {~T: ('a', <class 'int'>), ~U: ('b', <class 'str'>)}
    >>>
    >>> class MyStruct(Generic[T, U]):
    ...     a: T
    ...
    ...     @property
    ...     def myprop(self) -> U: ...
    >>> get_bound_typevars(MyStruct[float, bytes])
    {~T: ('a', <class 'float'>), ~U: ('myprop', <class 'bytes'>)}

    """
    origin = get_origin(obj) or obj
    hints = get_type_hints(origin, include_properties=True)
    params = get_type_params(obj)

    result = {}
    for attr, typ in hints.items():
        if isinstance(typ, TypeVar):
            result[typ] = (attr, params[typ.__name__])
    return result


def evaluate_annotations(
    annots: dict[str, str],
    module_name: str,
    class_name: Optional[str] = None,
    best_effort: bool = False,
) -> dict[str, Any]:
    """Evaluate type annotations that are strings.

    Parameters
    ----------
    annots
        Type annotations to evaluate.
    module_name
        The name of the module that the annotations are defined in, hence
        providing global scope.
    class_name
        The name of the class that the annotations are defined in, hence
        providing Self type.
    best_effort
        Whether to ignore errors when evaluating type annotations.

    Returns
    -------
    Actual type hints.

    Examples
    --------
    >>> annots = {"a": "dict[str, float]", "b": "int"}
    >>> evaluate_annotations(annots, __name__)
    {'a': dict[str, float], 'b': <class 'int'>}

    """
    module = sys.modules.get(module_name, None)
    globalns = getattr(module, "__dict__", None)
    if class_name is None:
        localns = None
    else:
        localns = dict(Self=f"{module_name}.{class_name}")

    result = {}
    for k, v in annots.items():
        if isinstance(v, str):
            try:
                v = eval(v, globalns, localns)  # noqa: S307
            except NameError:
                if not best_effort:
                    raise
        result[k] = v

    return result


def format_typehint(typ: Any) -> str:
    if isinstance(typ, type):
        return typ.__name__
    elif isinstance(typ, TypeVar):
        if typ.__bound__ is None:
            return str(typ)
        else:
            return format_typehint(typ.__bound__)
    else:
        # remove the module name from the typehint, including generics
        return re.sub(r"(\w+\.)+", "", str(typ))

# 在泛型编程中，我们经常希望定义类似 class MyGeneric[T = int]: ... 的结构。PEP 696 正式支持了这种语法，但在旧版本 Python 中，泛型必须显式指定所有参数。
# 自动补全缺失的泛型参数：如果用户定义了 MyClass[str]，但该类有多个泛型参数（例如 [T, U]），它会自动将 U 填充为预定义的默认类型。
# 统一 API 体验：使得 Ibis 内部的复杂数据结构（如表达式节点）在使用时无需强制要求用户显式传入每一个可能的泛型参数，降低了 API 的使用门槛。
class DefaultTypeVars:
    """Enable using default type variables in generic classes (PEP-0696)."""

    __slots__ = ()

    def __class_getitem__(cls, params):
        params = params if isinstance(params, tuple) else (params,)
        pairs = zip_longest(params, cls.__parameters__)
        params = tuple(p.__default__ if t is None else t for t, p in pairs)
        return super().__class_getitem__(params)


class Sentinel(type):
    """Create type-annotable unique objects."""

    def __new__(cls, name, bases, namespace, **kwargs):
        if bases:
            raise TypeError("Sentinels cannot be subclassed")
        return super().__new__(cls, name, bases, namespace, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("Sentinels are not constructible")


class CoercionError(Exception): ...

# Coercible 是一个类型转换协议（Type Conversion Protocol）。它定义了一套标准，允许 Ibis 将用户传入的“非标准输入”自动转换为 Ibis 内部标准的数据类型或表达式节点。
# 在构建复杂的表达式树（AST）时，用户经常会传递 Python 原生类型（如 int, str, list），而 Ibis 需要将它们“强制（Coerce）”转换为 Ibis 的对象（如 Literal, Column）。
# 统一转换接口：强制所有支持自动转换的类实现 __coerce__ 方法。
# 实现“智能构造”：配合 coerced_to 模式，使得 Ibis 的 API 更加友好。例如，函数参数标记为 coerced_to(dt.DataType)，传入字符串 'int64' 时，系统会自动调用 DataType.__coerce__('int64') 得到合法的类型对象。
# 解耦类型逻辑：将“如何从原始数据构造对象”的逻辑封装在目标类型内部，而不是放在校验器或外部工厂函数中。
class Coercible(Abstract):
    """Protocol for defining coercible types.

    Coercible types define a special `__coerce__` method that accepts an object
    with an instance of the type. Used in conjunction with the `coerced_to``
    pattern to coerce arguments to a specific type.
    """
    # cls: 目标类型本身（例如 DataType 或 Schema）。
    # value: Any: 待转换的原始输入（例如用户传入的 str 或 dict）。
    # / (位置参数限制符): 强制 value 必须作为位置参数传入，确保 API 调用的规范性。
    @classmethod
    @abstractmethod
    def __coerce__(cls, value: Any, /, **kwargs: Any) -> Self: ...


def get_defining_frame(obj):
    """Locate the outermost frame where `obj` is defined."""
    for frame_info in inspect.stack()[::-1]:
        for var in frame_info.frame.f_locals.values():
            if obj is var:
                return frame_info.frame
    raise ValueError(f"No defining frame found for {obj}")


def get_defining_scope(obj, types=None):
    """Get variables in the scope where `expr` is first defined."""
    frame = get_defining_frame(obj)
    scope = {**frame.f_globals, **frame.f_locals}
    if types is not None:
        scope = {k: v for k, v in scope.items() if isinstance(v, types)}
    return scope
