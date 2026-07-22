from __future__ import annotations

import math
import numbers
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from inspect import Parameter
from typing import (
    Annotated,
    ForwardRef,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)
from typing import Any as AnyType

import toolz
from typing_extensions import GenericMeta

from ibis.common.bases import FrozenSlotted as Slotted
from ibis.common.bases import Hashable, Singleton
from ibis.common.collections import FrozenDict, RewindableIterator, frozendict
from ibis.common.deferred import (
    Deferred,
    Factory,
    Resolver,
    Variable,
    _,  # noqa: F401
    resolver,
)
from ibis.common.typing import (
    Coercible,
    CoercionError,
    Sentinel,
    UnionType,
    _ClassInfo,
    format_typehint,
    get_bound_typevars,
    get_type_params,
)
from ibis.util import import_object, is_iterable, promote_list, unalias_package

T_co = TypeVar("T_co", covariant=True)


def as_resolver(obj):
    if callable(obj) and not isinstance(obj, Deferred):
        return Factory(obj)
    else:
        return resolver(obj)


class NoMatch(metaclass=Sentinel):
    """Marker to indicate that a pattern didn't match."""

# Ibis 类型校验、类型转换（Coercion）以及节点模式匹配（Pattern Matching/IR Rewriting） 系统的底层基石
# Pattern 是所有模式对象的抽象基类，主要承担三大核心职责：
# 类型校验与自动化转换：在构建 Ibis 表达式或节点（Node）时，系统需要校验用户传入的参数类型。Pattern 负责验证数据是否符合预期形态，并在必要时自动调用强制转换（Coercion）。
# 表达式图的模式匹配与重写（IR Rewriting）：Ibis 的编译器会对表达式树（IR）进行结构分析与优化。Pattern 提供了针对表达式树进行结构匹配、捕获（Capture）和替换（Replace）的能力。
# 基于 Python Typehint 快速构建规则：内置强大的元编程解析逻辑（from_typehint），能够将 Python 原生的类型注解（如 Union、Annotated、Tuple 等）自动转换为内部对应的 Pattern 验证器。
# TODO(kszucs): have an As[int] or Coerced[int] type in ibis.common.typing which
# would be used to annotate an argument as coercible to int or to a certain type
# without needing for the type to inherit from Coercible
class Pattern(Hashable):
    """Base class for all patterns.

    Patterns are used to match values against a given condition. They are extensively
    used by other core components of Ibis to validate and/or coerce user inputs.
    """
    # 解析 Python 标准类型注解（Typehint），将其递归映射并构造为对应的 Ibis Pattern 验证对象。
    # 主要任务是将 Python 原生的各种类型注解（Typehint）拆解，递归翻译为 Ibis 内部的 Pattern 校验/强转对象。
    # annot：被解析的类型注解（如 int, Optional[str], Annotated[int, Positive] 等）
    # 例如 name: str = "Alice"，这里的str就是类型注解
    # tags: list[str] = ["python", "ibis", "sql"]  list[str】也是
    # allow_coercion：布尔值，标识是否允许自动类型强转（若为 True，遇到实现 Coercible 协议的类型时会返回强转模式 CoercedTo，而非单纯的类型检查 InstanceOf）。
    @classmethod
    def from_typehint(cls, annot: type, allow_coercion: bool = True) -> Pattern:
        """Construct a validator from a python type annotation.

        Parameters
        ----------
        annot
            The typehint annotation to construct the pattern from. This must be
            an already evaluated type annotation.
        allow_coercion
            Whether to use coercion if the typehint is a Coercible type.

        Returns
        -------
        A pattern that matches the given type annotation.

        """
        # TODO(kszucs): cache the result of this function
        # TODO(kszucs): explore issubclass(typ, SupportsInt) etc.
        # 利用 Python typing 模块的底层函数提取泛型的“原始类型”（origin）与“泛型参数列表”（args）。
        # 示例：对于 list[int]，origin 是 list，args 是 (int,)。
        # 示例：对于非泛型 int，origin 为 None，args 为 ()。
        origin, args = get_origin(annot), get_args(annot)
        # 分支一：非泛型类型处理（origin is None
        if origin is None:
            # the typehint is not generic
            # 如果注解是 ...（Ellipsis）或者 Any（通配类型），直接返回匹配任意值的通配符模式 _any。
            if annot is Ellipsis or annot is AnyType:
                # treat both `Any` and `...` as wildcard
                return _any
            # 如果注解是一个具体的 Python 类（如 int、str 或 Ibis 自定义类 Table）
            elif isinstance(annot, type):
                # the typehint is a concrete type (e.g. int, str, etc.)
                # 若开启了 allow_coercion 且该类实现了 Coercible 强转协议，返回 CoercedTo(annot)（尝试强转）；
                if allow_coercion and issubclass(annot, Coercible):
                    # the type implements the Coercible protocol so we try to
                    # coerce the value to the given type rather than checking
                    return CoercedTo(annot)
                else:
                    # 否则返回 InstanceOf(annot)（仅做 isinstance 校验）。
                    return InstanceOf(annot)
            # 处理泛型类型变量 TypeVar（例如 T = TypeVar("T", bound=int)）：
            # 检查是否为协变（covariant），非协变目前抛出未实现异常；
            # 如果 TypeVar 指定了上界 bound（如绑定为 int），递归调用自身解析上界类型；否则返回通配符 _any。
            elif isinstance(annot, TypeVar):
                # if the typehint is a type variable we try to construct a
                # validator from it only if it is covariant and has a bound
                if not annot.__covariant__:
                    raise NotImplementedError(
                        "Only covariant typevars are supported for now"
                    )
                if annot.__bound__:
                    return cls.from_typehint(annot.__bound__)
                else:
                    return _any
            # 如果注解是 Enum 字段，返回要求值必须等于该枚举对象的 EqualTo 模式。
            elif isinstance(annot, Enum):
                # for enums we check the value against the enum values
                return EqualTo(annot)
            # 处理字符串形式的类型声明（如 'Table'）或前向引用 ForwardRef。生成 LazyInstanceOf 模式，在运行时再延迟查找对应的类。
            elif isinstance(annot, str):
                # for strings and forward references we check in a lazy way
                return LazyInstanceOf(annot)
            elif isinstance(annot, ForwardRef):
                return LazyInstanceOf(annot.__forward_arg__)
            else:
                raise TypeError(f"Cannot create validator from annotation {annot!r}")
        # 分支二：各种高级泛型类型处理（origin is not None）
        # 若显式注解了 CoercedTo[T]，直接提取内部第一个参数生成 CoercedTo 模式。
        elif origin is CoercedTo:
            return CoercedTo(args[0])
        # 处理字面量枚举（如 Literal["a", "b"]），返回 IsIn(args) 模式，校验输入值是否在指定的集合列表中。
        elif origin is Literal:
            # for literal types we check the value against the literal values
            return IsIn(args)
        # 联合类型：Union 与 Optional
        elif origin is UnionType or origin is Union:
            # this is slightly more complicated because we need to handle
            # Optional[T] which is Union[T, None] and Union[T1, T2, ...]
            # 拆解参数 args。如果最后一个参数是 type(None)（即 NoneType），说明它是一个 Optional[...]：
            *rest, last = args
            if last is type(None):
                # the typehint is Optional[*rest] which is equivalent to
                # Union[*rest, None], so we construct an Option pattern
                # 提取非空类型并递归解析为 inner 模式，最终外层包装为 Option(inner) 模式（允许为 None）。
                if len(rest) == 1:
                    inner = cls.from_typehint(rest[0])
                # 若不是 Optional，则将所有联合分支类型递归解析，包装为多选一模式 AnyOf。
                else:
                    inner = AnyOf(*map(cls.from_typehint, rest))
                return Option(inner)
            else:
                # the typehint is Union[*args] so we construct an AnyOf pattern
                return AnyOf(*map(cls.from_typehint, args))
        # 附加元数据：Annotated
        # 第一个参数 annot 是基础类型，其余参数 extras 是附加约束；
        elif origin is Annotated:
            # the Annotated typehint can be used to add extra validation logic
            # to the typehint, e.g. Annotated[int, Positive], the first argument
            # is used for isinstance checks, the rest are applied in conjunction
            annot, *extras = args
            # 递归解析 annot 并与 extras 组合，返回必须同时满足的 AllOf 模式。
            return AllOf(cls.from_typehint(annot), *extras)
        # 可调用对象：Callable
        elif origin is Callable:
            # the Callable typehint is used to annotate functions, e.g. the
            # following typehint annotates a function that takes two integers
            # and returns a string: Callable[[int, int], str]
            # 若带签名参数（如 Callable[[int], str]），提取参数列表和返回值，分别递归解析为模式，构造 CallableWith 模式；
            if args:
                # callable with args and return typehints construct a special
                # CallableWith validator
                arg_hints, return_hint = args
                arg_patterns = tuple(map(cls.from_typehint, arg_hints))
                return_pattern = cls.from_typehint(return_hint)
                return CallableWith(arg_patterns, return_pattern)
            else:
                # in case of Callable without args we check for the Callable
                # protocol only
                # 若未指定签名（如单纯的 Callable），只做 InstanceOf(Callable) 校验。
                return InstanceOf(Callable)
        # 处理 tuple 注解：
        elif issubclass(origin, tuple):
            # construct validators for the tuple elements, but need to treat
            # variadic tuples differently, e.g. tuple[int, ...] is a variadic
            # tuple of integers, while tuple[int] is a tuple with a single int
            # 变长元组（如 tuple[int, ...]）：rest 匹配到 Ellipsis，解析 first 类型并构造 TupleOf 模式；
            first, *rest = args
            if rest == [Ellipsis]:
                return TupleOf(cls.from_typehint(first))
            else:
                # 定长元组（如 tuple[int, str]）：将每个位置的类型分别解析，构造定长列表模式 PatternList。
                return PatternList(map(cls.from_typehint, args), type=origin)
        # 序列与字典：Sequence & Mapping
        # 处理列表、序列类容器（如 Sequence[int] 或 list[str]）：
        elif issubclass(origin, Sequence):
            # construct a validator for the sequence elements where all elements
            # must be of the same type, e.g. Sequence[int] is a sequence of ints
            # 解析内部元素类型 value_inner；
            (value_inner,) = map(cls.from_typehint, args)
            if allow_coercion and issubclass(origin, Coercible):
                return GenericSequenceOf(value_inner, type=origin)
            else:
                return SequenceOf(value_inner, type=origin)
        elif issubclass(origin, Mapping):
            # construct a validator for the mapping keys and values, e.g.
            # Mapping[str, int] is a mapping with string keys and int values
            key_inner, value_inner = map(cls.from_typehint, args)
            return MappingOf(key_inner, value_inner, type=origin)
        # 通用泛型类：GenericMeta
        # 根据强转许可与 Coercible 判定，返回对应的 GenericCoercedTo 或 GenericInstanceOf。
        elif isinstance(origin, GenericMeta):
            # construct a validator for the generic type, see the specific
            # Generic* validators for more details
            if allow_coercion and issubclass(origin, Coercible) and args:
                return GenericCoercedTo(annot)
            else:
                return GenericInstanceOf(annot)
        else:
            raise TypeError(
                f"Cannot create validator from annotation {annot!r} {origin!r}"
            )
    # Pattern 自身是一个抽象基类，不直接执行具体的匹配逻辑。所有继承 Pattern 的子类（如 InstanceOf、AnyOf、CoercedTo、SequenceOf 等）必须实现并重写此方法。
    # value: AnyType 待匹配、校验或强转的具体数值或对象。
    # 基础数据/Python 对象：例如 123、"hello"、[1, 2, 3] 等。
    # Ibis IR 表达式节点：例如 Literal 节点、Column 节点、或者整个操作节点 Node（在进行 IR 树重写和模式匹配时）。
    # context: dict[str, AnyType] 匹配过程中共享的上下文状态字典（Context Dictionary）
    # 变量捕获（Capture）：当模式中包含捕获表达式（例如 "x" @ PatternA）时，匹配成功的 value 会以键名 "x" 保存到 context 字典中，供后续的重写逻辑或谓词检查使用。
    # 跨节点状态共享：在递归匹配复杂的深层数据结构（如嵌套列表或 IR 语法树）时，context 作为“记忆库”在各个节点的 match 方法之间传递。
    @abstractmethod
    def match(self, value: AnyType, context: dict[str, AnyType]) -> AnyType:
        """Match a value against the pattern.

        Parameters
        ----------
        value
            The value to match the pattern against.
        context
            A dictionary providing arbitrary context for the pattern matching.

        Returns
        -------
        The result of the pattern matching. If the pattern doesn't match
        the value, then it must return the `NoMatch` sentinel value.

        """
        ...
    # 生成当前 Pattern 的可读文本描述（通常用于生成清晰的报错信息或调试日志）。
    def describe(self, plural=False):
        return f"matching {self!r}"
    # 判断两个 Pattern 对象在逻辑上是否等价。
    # 由于子类需要被缓存或进行去重比对，子类必须实现此相等性判断。
    @abstractmethod
    def __eq__(self, other: Pattern) -> bool: ...
    # 获取 Pattern 对象的哈希值。
    # 搭配基类声明的 Hashable 接口，确保 Pattern 实例可以作为字典的键（Key）或存入集合（Set）中。
    @abstractmethod
    def __hash__(self) -> int: ...
    # 魔法方法（重载取反运算符 ~）
    # 生成取反模式（逻辑非）。
    # 用法示例：~PatternA 会返回一个 Not(PatternA) 实例，匹配所有不符合 PatternA 的值。
    def __invert__(self) -> Not:
        """Syntax sugar for matching the inverse of the pattern."""
        return Not(self)
    # 魔法方法（重载位或运算符 |）
    # 生成析取模式（逻辑或）。
    # 用法示例：PatternA | PatternB 会返回一个 AnyOf(PatternA, PatternB) 实例，值只需满足其中任意一个模式即可匹配。
    def __or__(self, other: Pattern) -> AnyOf:
        """Syntax sugar for matching either of the patterns.

        Parameters
        ----------
        other
            The other pattern to match against.

        Returns
        -------
        New pattern that matches if either of the patterns match.

        """
        return AnyOf(self, other)
    # 魔法方法（重载位与运算符 &）
    # 生成合取模式（逻辑与）。
    # PatternA & PatternB 会返回一个 AllOf(PatternA, PatternB) 实例，值必须同时满足这两个模式。
    def __and__(self, other: Pattern) -> AllOf:
        """Syntax sugar for matching both of the patterns.

        Parameters
        ----------
        other
            The other pattern to match against.

        Returns
        -------
        New pattern that matches if both of the patterns match.

        """
        return AllOf(self, other)
    # 魔法方法（重载右移运算符 >>）
    # 语法糖，构建“匹配并替换”操作。
    # 用法示例：PatternA >> deferred_func 会返回 Replace(PatternA, deferred_func)。当 PatternA 匹配成功时，将值替换为计算后的新值，常用于 IR 树重写。
    def __rshift__(self, other: Deferred) -> Replace:
        """Syntax sugar for replacing a value.

        Parameters
        ----------
        other
            The deferred to use for constructing the replacement value.

        Returns
        -------
        New replace pattern.

        """
        return Replace(self, other)
    # 魔法方法（重载反向 @ 运算符）
    # 语法糖，给模式绑定标识符名称，将匹配到的对象“捕获”到上下文中。
    # 用法示例："x" @ PatternA 触发 PatternA.__rmatmul__("x")，返回 Capture("x", PatternA)，匹配成功后将匹配值以键 "x" 存入 context。
    def __rmatmul__(self, name: str) -> Capture:
        """Syntax sugar for capturing a value.

        Parameters
        ----------
        name
            The name of the capture.

        Returns
        -------
        New capture pattern.

        """
        return Capture(name, self)
    # 魔法方法（重载迭代接口 iter()）
    # 生成可重复匹配的序列子模式（SomeOf）。
    def __iter__(self) -> SomeOf:
        yield SomeOf(self)


class Is(Slotted, Pattern):
    """Pattern that matches a value against a reference value.

    Parameters
    ----------
    value
        The reference value to match against.

    """

    __slots__ = ("value",)
    value: AnyType

    def match(self, value, context):
        if value is self.value:
            return value
        else:
            return NoMatch


class Any(Slotted, Singleton, Pattern):
    """Pattern that accepts any value, basically a no-op."""

    def match(self, value, context):
        return value


_any = Any()


class Nothing(Slotted, Singleton, Pattern):
    """Pattern that no values."""

    def match(self, value, context):
        return NoMatch


class Capture(Slotted, Pattern):
    """Pattern that captures a value in the context.

    Parameters
    ----------
    pattern
        The pattern to match against.
    key
        The key to use in the context if the pattern matches.

    """

    __slots__ = ("key", "pattern")
    key: AnyType
    pattern: Pattern

    def __init__(self, key, pat=_any):
        if isinstance(key, (Deferred, Resolver)):
            key = as_resolver(key)
            if isinstance(key, Variable):
                key = key.name
            else:
                raise TypeError("Only variables can be used as capture keys")
        super().__init__(key=key, pattern=pattern(pat))

    def match(self, value, context):
        value = self.pattern.match(value, context)
        if value is NoMatch:
            return NoMatch
        context[self.key] = value
        return value


class Replace(Slotted, Pattern):
    """Pattern that replaces a value with the output of another pattern.

    Parameters
    ----------
    matcher
        The pattern to match against.
    replacer
        The deferred to use as a replacement.

    """

    __slots__ = ("matcher", "replacer")
    matcher: Pattern
    replacer: Resolver

    def __init__(self, matcher, replacer):
        super().__init__(matcher=pattern(matcher), replacer=as_resolver(replacer))

    def match(self, value, context):
        value = self.matcher.match(value, context)
        if value is NoMatch:
            return NoMatch
        # use the `_` reserved variable to record the value being replaced
        # in the context, so that it can be used in the replacer pattern
        context["_"] = value
        return self.replacer.resolve(context)


def replace(matcher):
    """More convenient syntax for replacing a value with the output of a function."""

    def decorator(replacer):
        return Replace(matcher, replacer)

    return decorator


class Check(Slotted, Pattern):
    """Pattern that checks a value against a predicate.

    Parameters
    ----------
    predicate
        The predicate to use.

    """

    __slots__ = ("predicate",)
    predicate: Callable

    @classmethod
    def __create__(cls, predicate):
        if isinstance(predicate, (Deferred, Resolver)):
            return DeferredCheck(predicate)
        else:
            return super().__create__(predicate)

    def __init__(self, predicate):
        assert callable(predicate)
        super().__init__(predicate=predicate)

    def describe(self, plural=False):
        if plural:
            return f"values that satisfy {self.predicate.__name__}()"
        else:
            return f"a value that satisfies {self.predicate.__name__}()"

    def match(self, value, context):
        if self.predicate(value):
            return value
        else:
            return NoMatch


class DeferredCheck(Slotted, Pattern):
    __slots__ = ("resolver",)
    resolver: Resolver

    def __init__(self, obj):
        super().__init__(resolver=as_resolver(obj))

    def describe(self, plural=False):
        if plural:
            return f"values that satisfy {self.resolver!r}"
        else:
            return f"a value that satisfies {self.resolver!r}"

    def match(self, value, context):
        context["_"] = value
        if self.resolver.resolve(context):
            return value
        else:
            return NoMatch


class Custom(Slotted, Pattern):
    """User defined custom matcher function.

    Parameters
    ----------
    func
        The function to apply.

    """

    __slots__ = ("func",)
    func: Callable

    def __init__(self, func):
        assert callable(func)
        super().__init__(func=func)

    def match(self, value, context):
        return self.func(value, context)


class EqualTo(Slotted, Pattern):
    """Pattern that checks a value equals to the given value.

    Parameters
    ----------
    value
        The value to check against.

    """

    __slots__ = ("value",)
    value: AnyType

    @classmethod
    def __create__(cls, value):
        if isinstance(value, (Deferred, Resolver)):
            return DeferredEqualTo(value)
        else:
            return super().__create__(value)

    def __init__(self, value):
        super().__init__(value=value)

    def match(self, value, context):
        if value == self.value:
            return value
        else:
            return NoMatch

    def describe(self, plural=False):
        return repr(self.value)


class DeferredEqualTo(Slotted, Pattern):
    """Pattern that checks a value equals to the given value.

    Parameters
    ----------
    value
        The value to check against.

    """

    __slots__ = ("resolver",)
    resolver: Resolver

    def __init__(self, obj):
        super().__init__(resolver=as_resolver(obj))

    def match(self, value, context):
        context["_"] = value
        if value == self.resolver.resolve(context):
            return value
        else:
            return NoMatch

    def describe(self, plural=False):
        return repr(self.resolver)


class Option(Slotted, Pattern):
    """Pattern that matches `None` or a value that passes the inner validator.

    Parameters
    ----------
    pattern
        The inner pattern to use.

    """

    __slots__ = ("default", "pattern")
    pattern: Pattern
    default: AnyType

    def __init__(self, pat, default=None):
        super().__init__(pattern=pattern(pat), default=default)

    def describe(self, plural=False):
        if plural:
            return f"optional {self.pattern.describe(plural=True)}"
        else:
            return f"either None or {self.pattern.describe(plural=False)}"

    def match(self, value, context):
        if value is None:
            if self.default is None:
                return None
            else:
                return self.default
        else:
            return self.pattern.match(value, context)


def _describe_type(typ, plural=False):
    if isinstance(typ, tuple):
        *rest, last = typ
        rest = ", ".join(_describe_type(t, plural=plural) for t in rest)
        last = _describe_type(last, plural=plural)
        return f"{rest} or {last}" if rest else last

    name = format_typehint(typ)
    if plural:
        return f"{name}s"
    elif name[0].lower() in "aeiou":
        return f"an {name}"
    else:
        return f"a {name}"


class TypeOf(Slotted, Pattern):
    """Pattern that matches a value that is of a given type."""

    __slots__ = ("type",)
    type: type

    def __init__(self, typ):
        super().__init__(type=typ)

    def describe(self, plural=False):
        return f"exactly {_describe_type(self.type, plural=plural)}"

    def match(self, value, context):
        if type(value) is self.type:
            return value
        else:
            return NoMatch


class SubclassOf(Slotted, Pattern):
    """Pattern that matches a value that is a subclass of a given type.

    Parameters
    ----------
    type
        The type to check against.

    """

    __slots__ = ("type",)

    def __init__(self, typ):
        super().__init__(type=typ)

    def describe(self, plural=False):
        if plural:
            return f"subclasses of {self.type.__name__}"
        else:
            return f"a subclass of {self.type.__name__}"

    def match(self, value, context):
        if issubclass(value, self.type):
            return value
        else:
            return NoMatch


class InstanceOf(Slotted, Singleton, Pattern):
    """Pattern that matches a value that is an instance of a given type.

    Parameters
    ----------
    types
        The type to check against.

    """

    __slots__ = ("type",)
    type: _ClassInfo

    def __init__(self, typ: type | tuple[type, ...]) -> None:
        super().__init__(type=typ)

    def __eq__(self, other: Pattern) -> bool:
        return type(other) is type(self) and frozenset(
            promote_list(self.type)
        ) == frozenset(promote_list(other.type))

    def __hash__(self) -> int:
        return super().__hash__()

    def describe(self, plural=False):
        return _describe_type(self.type, plural=plural)

    def match(self, value, context):
        if isinstance(value, self.type):
            return value
        else:
            return NoMatch

    def __call__(self, *args, **kwargs):
        return Object(self.type, *args, **kwargs)


class GenericInstanceOf(Slotted, Pattern):
    """Pattern that matches a value that is an instance of a given generic type.

    Parameters
    ----------
    typ
        The type to check against (must be a generic type).

    Examples
    --------
    >>> class MyNumber(Generic[T_co]):
    ...     value: T_co
    ...
    ...     def __init__(self, value: T_co):
    ...         self.value = value
    ...
    ...     def __eq__(self, other):
    ...         return type(self) is type(other) and self.value == other.value
    >>> p = GenericInstanceOf(MyNumber[int])
    >>> assert p.match(MyNumber(1), {}) == MyNumber(1)
    >>> assert p.match(MyNumber(1.0), {}) is NoMatch
    >>>
    >>> p = GenericInstanceOf(MyNumber[float])
    >>> assert p.match(MyNumber(1.0), {}) == MyNumber(1.0)
    >>> assert p.match(MyNumber(1), {}) is NoMatch

    """

    __slots__ = ("fields", "origin", "type")
    origin: type
    fields: FrozenDict[str, Pattern]

    def __init__(self, typ):
        origin = get_origin(typ)
        typevars = get_bound_typevars(typ)

        fields = {}
        for var, (attr, type_) in typevars.items():
            if not var.__covariant__:
                raise TypeError(
                    f"Typevar {var} is not covariant, cannot use it in a GenericInstanceOf"
                )
            fields[attr] = Pattern.from_typehint(type_, allow_coercion=False)

        super().__init__(type=typ, origin=origin, fields=frozendict(fields))

    def describe(self, plural=False):
        return _describe_type(self.type, plural=plural)

    def match(self, value, context):
        if not isinstance(value, self.origin):
            return NoMatch

        for name, pattern in self.fields.items():
            attr = getattr(value, name)
            if pattern.match(attr, context) is NoMatch:
                return NoMatch

        return value


class LazyInstanceOf(Slotted, Pattern):
    """A version of `InstanceOf` that accepts qualnames instead of imported classes.

    Useful for delaying imports.

    Parameters
    ----------
    types
        The types to check against.

    """

    __fields__ = ("qualname", "package")
    __slots__ = ("loaded", "package", "qualname")
    qualname: str
    package: str
    loaded: type

    def __init__(self, qualname):
        package = unalias_package(qualname.split(".", 1)[0])
        super().__init__(qualname=qualname, package=package)

    def match(self, value, context):
        if hasattr(self, "loaded"):
            return value if isinstance(value, self.loaded) else NoMatch

        for klass in type(value).__mro__:
            package = klass.__module__.split(".", 1)[0]
            if package == self.package:
                typ = import_object(self.qualname)
                object.__setattr__(self, "loaded", typ)
                return value if isinstance(value, typ) else NoMatch

        return NoMatch


class CoercedTo(Slotted, Pattern, Generic[T_co]):
    """Force a value to have a particular Python type.

    If a Coercible subclass is passed, the `__coerce__` method will be used to
    coerce the value. Otherwise, the type will be called with the value as the
    only argument.

    Parameters
    ----------
    type
        The type to coerce to.

    """

    __slots__ = ("func", "type")
    type: T_co

    def __init__(self, type):
        func = type.__coerce__ if issubclass(type, Coercible) else type
        super().__init__(type=type, func=func)

    def describe(self, plural=False):
        type = _describe_type(self.type, plural=False)
        if plural:
            return f"coercibles to {type}"
        else:
            return f"coercible to {type}"

    def match(self, value, context):
        try:
            value = self.func(value)
        except (TypeError, CoercionError):
            return NoMatch

        if isinstance(value, self.type):
            return value
        else:
            return NoMatch

    def __call__(self, *args, **kwargs):
        return Object(self.type, *args, **kwargs)


class GenericCoercedTo(Slotted, Pattern):
    """Force a value to have a particular generic Python type.

    Parameters
    ----------
    typ
        The type to coerce to. Must be a generic type with bound typevars.

    Examples
    --------
    >>> from typing import Generic, TypeVar
    >>>
    >>> T = TypeVar("T", covariant=True)
    >>>
    >>> class MyNumber(Coercible, Generic[T]):
    ...     __slots__ = ("value",)
    ...
    ...     def __init__(self, value):
    ...         self.value = value
    ...
    ...     def __eq__(self, other):
    ...         return type(self) is type(other) and self.value == other.value
    ...
    ...     @classmethod
    ...     def __coerce__(cls, value, T=None):
    ...         if issubclass(T, int):
    ...             return cls(int(value))
    ...         elif issubclass(T, float):
    ...             return cls(float(value))
    ...         else:
    ...             raise CoercionError(f"Cannot coerce to {T}")
    >>> p = GenericCoercedTo(MyNumber[int])
    >>> assert p.match(3.14, {}) == MyNumber(3)
    >>> assert p.match("15", {}) == MyNumber(15)
    >>>
    >>> p = GenericCoercedTo(MyNumber[float])
    >>> assert p.match(3.14, {}) == MyNumber(3.14)
    >>> assert p.match("15", {}) == MyNumber(15.0)

    """

    __slots__ = ("checker", "origin", "params")
    origin: type
    params: FrozenDict[str, type]
    checker: GenericInstanceOf

    def __init__(self, target):
        origin = get_origin(target)
        checker = GenericInstanceOf(target)
        params = frozendict(get_type_params(target))
        super().__init__(origin=origin, params=params, checker=checker)

    def describe(self, plural=False):
        if plural:
            return f"coercibles to {self.checker.describe(plural=False)}"
        else:
            return f"coercible to {self.checker.describe(plural=False)}"

    def match(self, value, context):
        try:
            value = self.origin.__coerce__(value, **self.params)
        except CoercionError:
            return NoMatch

        if self.checker.match(value, context) is NoMatch:
            return NoMatch

        return value


class Not(Slotted, Pattern):
    """Pattern that matches a value that does not match a given pattern.

    Parameters
    ----------
    pattern
        The pattern which the value should not match.

    """

    __slots__ = ("pattern",)
    pattern: Pattern

    def __init__(self, inner):
        super().__init__(pattern=pattern(inner))

    def describe(self, plural=False):
        if plural:
            return f"anything except {self.pattern.describe(plural=True)}"
        else:
            return f"anything except {self.pattern.describe(plural=False)}"

    def match(self, value, context):
        if self.pattern.match(value, context) is NoMatch:
            return value
        else:
            return NoMatch


class AnyOf(Slotted, Pattern):
    """Pattern that if any of the given patterns match.

    Parameters
    ----------
    patterns
        The patterns to match against. The first pattern that matches will be
        returned.

    """

    __slots__ = ("patterns",)
    patterns: tuple[Pattern, ...]

    def __init__(self, *patterns: Pattern) -> None:
        super().__init__(patterns=tuple(map(pattern, patterns)))

    def __eq__(self, other: Pattern) -> bool:
        return type(self) is type(other) and frozenset(self.patterns) == frozenset(
            other.patterns
        )

    def __hash__(self) -> int:
        return super().__hash__()

    def describe(self, plural=False):
        *rest, last = self.patterns
        rest = ", ".join(p.describe(plural=plural) for p in rest)
        last = last.describe(plural=plural)
        return f"{rest} or {last}" if rest else last

    def match(self, value, context):
        for pattern in self.patterns:
            result = pattern.match(value, context)
            if result is not NoMatch:
                return result
        return NoMatch


class AllOf(Slotted, Pattern):
    """Pattern that matches if all of the given patterns match.

    Parameters
    ----------
    patterns
        The patterns to match against. The value will be passed through each
        pattern in order. The changes applied to the value propagate through the
        patterns.

    """

    __slots__ = ("patterns",)
    patterns: tuple[Pattern, ...]

    def __init__(self, *pats):
        patterns = tuple(map(pattern, pats))
        super().__init__(patterns=patterns)

    def describe(self, plural=False):
        *rest, last = self.patterns
        rest = ", ".join(p.describe(plural=plural) for p in rest)
        last = last.describe(plural=plural)
        return f"{rest} then {last}" if rest else last

    def match(self, value, context):
        for pattern in self.patterns:
            value = pattern.match(value, context)
            if value is NoMatch:
                return NoMatch
        return value


class Length(Slotted, Pattern):
    """Pattern that matches if the length of a value is within a given range.

    Parameters
    ----------
    exactly
        The exact length of the value. If specified, `at_least` and `at_most`
        must be None.
    at_least
        The minimum length of the value.
    at_most
        The maximum length of the value.

    """

    __slots__ = ("at_least", "at_most")
    at_least: int
    at_most: int

    def __init__(
        self,
        exactly: Optional[int] = None,
        at_least: Optional[int] = None,
        at_most: Optional[int] = None,
    ):
        if exactly is not None:
            if at_least is not None or at_most is not None:
                raise ValueError("Can't specify both exactly and at_least/at_most")
            at_least = exactly
            at_most = exactly
        super().__init__(at_least=at_least, at_most=at_most)

    def describe(self, plural=False):
        if self.at_least is not None and self.at_most is not None:
            if self.at_least == self.at_most:
                return f"with length exactly {self.at_least}"
            else:
                return f"with length between {self.at_least} and {self.at_most}"
        elif self.at_least is not None:
            return f"with length at least {self.at_least}"
        elif self.at_most is not None:
            return f"with length at most {self.at_most}"
        else:
            return "with any length"

    def match(self, value, context):
        length = len(value)
        if self.at_least is not None and length < self.at_least:
            return NoMatch
        if self.at_most is not None and length > self.at_most:
            return NoMatch
        return value


class Between(Slotted, Pattern):
    """Match a value between two bounds.

    Parameters
    ----------
    lower
        The lower bound.
    upper
        The upper bound.

    """

    __slots__ = ("lower", "upper")
    lower: float
    upper: float

    def __init__(self, lower: float = -math.inf, upper: float = math.inf):
        super().__init__(lower=lower, upper=upper)

    def match(self, value, context):
        if self.lower <= value <= self.upper:
            return value
        else:
            return NoMatch


class Contains(Slotted, Pattern):
    """Pattern that matches if a value contains a given value.

    Parameters
    ----------
    needle
        The item that the passed value should contain.

    """

    __slots__ = ("needle",)
    needle: AnyType

    def __init__(self, needle):
        super().__init__(needle=needle)

    def describe(self, plural=False):
        return f"containing {self.needle!r}"

    def match(self, value, context):
        if self.needle in value:
            return value
        else:
            return NoMatch


class IsIn(Slotted, Pattern):
    """Pattern that matches if a value is in a given set.

    Parameters
    ----------
    haystack
        The set of values that the passed value should be in.

    """

    __slots__ = ("haystack",)
    haystack: frozenset

    def __init__(self, haystack):
        super().__init__(haystack=frozenset(haystack))

    def describe(self, plural=False):
        return f"in {set(self.haystack)!r}"

    def match(self, value, context):
        if value in self.haystack:
            return value
        else:
            return NoMatch


class SequenceOf(Slotted, Pattern):
    """Pattern that matches if all of the items in a sequence match a given pattern.

    Specialization of the more flexible GenericSequenceOf pattern which uses two
    additional patterns to possibly coerce the sequence type and to match on
    the length of the sequence.

    Parameters
    ----------
    item
        The pattern to match against each item in the sequence.
    type
        The type to coerce the sequence to. Defaults to tuple.

    """

    __slots__ = ("item", "type")
    item: Pattern
    type: type

    def __init__(self, item, type=list):
        super().__init__(item=pattern(item), type=type)

    def describe(self, plural=False):
        typ = _describe_type(self.type, plural=plural)
        item = self.item.describe(plural=True)
        return f"{typ} of {item}"

    def match(self, values, context):
        if not is_iterable(values):
            return NoMatch

        if self.item == _any:
            # optimization to avoid unnecessary iteration
            result = values
        else:
            result = []
            for item in values:
                item = self.item.match(item, context)
                if item is NoMatch:
                    return NoMatch
                result.append(item)

        return self.type(result)


class GenericSequenceOf(Slotted, Pattern):
    """Pattern that matches if all of the items in a sequence match a given pattern.

    Parameters
    ----------
    item
        The pattern to match against each item in the sequence.
    type
        The type to coerce the sequence to. Defaults to list.
    exactly
        The exact length of the sequence.
    at_least
        The minimum length of the sequence.
    at_most
        The maximum length of the sequence.

    """

    __slots__ = ("item", "length", "type")
    item: Pattern
    type: Pattern
    length: Length

    def __init__(
        self,
        item: Pattern,
        type: type = list,
        exactly: Optional[int] = None,
        at_least: Optional[int] = None,
        at_most: Optional[int] = None,
    ):
        item = pattern(item)
        type = CoercedTo(type)
        length = Length(exactly=exactly, at_least=at_least, at_most=at_most)
        super().__init__(item=item, type=type, length=length)

    def match(self, values, context):
        if not is_iterable(values):
            return NoMatch

        if self.item == _any:
            # optimization to avoid unnecessary iteration
            result = values
        else:
            result = []
            for value in values:
                value = self.item.match(value, context)
                if value is NoMatch:
                    return NoMatch
                result.append(value)

        result = self.type.match(result, context)
        if result is NoMatch:
            return NoMatch

        return self.length.match(result, context)


class GenericMappingOf(Slotted, Pattern):
    """Pattern that matches if all of the keys and values match the given patterns.

    Parameters
    ----------
    key
        The pattern to match the keys against.
    value
        The pattern to match the values against.
    type
        The type to coerce the mapping to. Defaults to dict.

    """

    __slots__ = ("key", "type", "value")
    key: Pattern
    value: Pattern
    type: Pattern

    def __init__(self, key: Pattern, value: Pattern, type: type = dict):
        super().__init__(key=pattern(key), value=pattern(value), type=CoercedTo(type))

    def match(self, value, context):
        if not isinstance(value, Mapping):
            return NoMatch

        result = {}
        for k, v in value.items():
            if (k := self.key.match(k, context)) is NoMatch:
                return NoMatch
            if (v := self.value.match(v, context)) is NoMatch:
                return NoMatch
            result[k] = v

        result = self.type.match(result, context)
        if result is NoMatch:
            return NoMatch

        return result


MappingOf = GenericMappingOf


class Attrs(Slotted, Pattern):
    __slots__ = ("fields",)
    fields: FrozenDict[str, Pattern]

    def __init__(self, **fields):
        fields = frozendict(toolz.valmap(pattern, fields))
        super().__init__(fields=fields)

    def match(self, value, context):
        for attr, pattern in self.fields.items():
            if not hasattr(value, attr):
                return NoMatch

            v = getattr(value, attr)
            if match(pattern, v, context) is NoMatch:
                return NoMatch

        return value


class Object(Slotted, Pattern):
    """Pattern that matches if the object has the given attributes and they match the given patterns.

    The type must conform the structural pattern matching protocol, e.g. it must have a
    __match_args__ attribute that is a tuple of the names of the attributes to match.

    Parameters
    ----------
    type
        The type of the object.
    *args
        The positional arguments to match against the attributes of the object.
    **kwargs
        The keyword arguments to match against the attributes of the object.

    """

    __slots__ = ("args", "kwargs", "type")
    type: Pattern
    args: tuple[Pattern, ...]
    kwargs: FrozenDict[str, Pattern]

    @classmethod
    def __create__(cls, type, *args, **kwargs):
        if not args and not kwargs:
            return InstanceOf(type)
        return super().__create__(type, *args, **kwargs)

    def __init__(self, typ, *args, **kwargs):
        if isinstance(typ, type) and len(typ.__match_args__) < len(args):
            raise ValueError(
                "The type to match has fewer `__match_args__` than the number "
                "of positional arguments in the pattern"
            )
        typ = pattern(typ)
        args = tuple(map(pattern, args))
        kwargs = frozendict(toolz.valmap(pattern, kwargs))
        super().__init__(type=typ, args=args, kwargs=kwargs)

    def match(self, value, context):
        if self.type.match(value, context) is NoMatch:
            return NoMatch

        # the pattern requirest more positional arguments than the object has
        if len(value.__match_args__) < len(self.args):
            return NoMatch
        patterns = dict(zip(value.__match_args__, self.args))
        patterns.update(self.kwargs)

        fields = {}
        changed = False
        for name, pattern in patterns.items():
            try:
                attr = getattr(value, name)
            except AttributeError:
                return NoMatch

            result = pattern.match(attr, context)
            if result is NoMatch:
                return NoMatch
            elif result != attr:
                changed = True
                fields[name] = result
            else:
                fields[name] = attr

        if changed:
            return type(value)(**fields)
        else:
            return value


class Node(Slotted, Pattern):
    __slots__ = ("each_arg", "type")
    type: Pattern

    def __init__(self, type, each_arg):
        super().__init__(type=pattern(type), each_arg=pattern(each_arg))

    def match(self, value, context):
        if self.type.match(value, context) is NoMatch:
            return NoMatch

        newargs = {}
        changed = False
        for name, arg in zip(value.__argnames__, value.__args__):
            result = self.each_arg.match(arg, context)
            if result is NoMatch:
                newargs[name] = arg
            else:
                newargs[name] = result
                changed = True

        if changed:
            return value.__class__(**newargs)
        else:
            return value


class CallableWith(Slotted, Pattern):
    __slots__ = ("args", "return_")
    args: tuple
    return_: AnyType

    def __init__(self, args, return_=_any):
        super().__init__(args=tuple(args), return_=return_)

    def match(self, value, context):
        from ibis.common.annotations import EMPTY, annotated

        if not callable(value):
            return NoMatch

        fn = annotated(self.args, self.return_, value)

        has_varargs = False
        positional, required_positional = [], []
        for p in fn.__signature__.parameters.values():
            if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
                positional.append(p)
                if p.default is EMPTY:
                    required_positional.append(p)
            elif p.kind is Parameter.KEYWORD_ONLY and p.default is EMPTY:
                raise TypeError(
                    "Callable has mandatory keyword-only arguments which cannot be specified"
                )
            elif p.kind is Parameter.VAR_POSITIONAL:
                has_varargs = True

        if len(required_positional) > len(self.args):
            # Callable has more positional arguments than expected")
            return NoMatch
        elif len(positional) < len(self.args) and not has_varargs:
            # Callable has less positional arguments than expected")
            return NoMatch
        else:
            return fn


class SomeOf(Slotted, Pattern):
    __slots__ = ("delimiter", "pattern")

    @classmethod
    def __create__(cls, *args, **kwargs):
        if len(args) == 1:
            return super().__create__(*args, **kwargs)
        else:
            return SomeChunksOf(*args, **kwargs)

    def __init__(self, item, **kwargs):
        pattern = GenericSequenceOf(item, **kwargs)
        delimiter = pattern.item
        super().__init__(pattern=pattern, delimiter=delimiter)

    def match(self, values, context):
        return self.pattern.match(values, context)


class SomeChunksOf(Slotted, Pattern):
    """Pattern that unpacks a value into its elements.

    Designed to be used inside a `PatternList` pattern with the `*` syntax.
    """

    __slots__ = ("delimiter", "pattern")

    def __init__(self, *args, **kwargs):
        pattern = GenericSequenceOf(PatternList(args), **kwargs)
        delimiter = pattern.item.patterns[0]
        super().__init__(pattern=pattern, delimiter=delimiter)

    def chunk(self, values, context):
        chunk = []
        for item in values:
            if self.delimiter.match(item, context) is NoMatch:
                chunk.append(item)
            else:
                if chunk:  # only yield if there are items in the chunk
                    yield chunk
                chunk = [item]  # start a new chunk with the delimiter
        if chunk:
            yield chunk

    def match(self, values, context):
        chunks = self.chunk(values, context)
        result = self.pattern.match(chunks, context)
        if result is NoMatch:
            return NoMatch
        else:
            return [el for lst in result for el in lst]


def _maybe_unwrap_capture(obj):
    return obj.pattern if isinstance(obj, Capture) else obj


class PatternList(Slotted, Pattern):
    """Pattern that matches if the respective items in a tuple match the given patterns.

    Parameters
    ----------
    fields
        The patterns to match the respective items in the tuple.

    """

    __slots__ = ("patterns", "type")
    patterns: tuple[Pattern, ...]
    type: type

    @classmethod
    def __create__(cls, patterns, type=list):
        if patterns == ():
            return EqualTo(patterns)

        patterns = tuple(map(pattern, patterns))
        for pat in patterns:
            pat = _maybe_unwrap_capture(pat)
            if isinstance(pat, (SomeOf, SomeChunksOf)):
                return VariadicPatternList(patterns, type)

        return super().__create__(patterns, type)

    def __init__(self, patterns, type):
        super().__init__(patterns=patterns, type=type)

    def describe(self, plural=False):
        patterns = ", ".join(f.describe(plural=False) for f in self.patterns)
        if plural:
            return f"tuples of ({patterns})"
        else:
            return f"a tuple of ({patterns})"

    def match(self, values, context):
        if not is_iterable(values):
            return NoMatch

        if len(values) != len(self.patterns):
            return NoMatch

        result = []
        for pattern, value in zip(self.patterns, values):
            value = pattern.match(value, context)
            if value is NoMatch:
                return NoMatch
            result.append(value)

        return self.type(result)


class VariadicPatternList(Slotted, Pattern):
    __slots__ = ("patterns", "type")
    patterns: tuple[Pattern, ...]
    type: type

    def __init__(self, patterns, type=list):
        patterns = tuple(map(pattern, patterns))
        super().__init__(patterns=patterns, type=type)

    def match(self, value, context):
        if not self.patterns:
            return NoMatch if value else []

        it = RewindableIterator(value)
        result = []

        following_patterns = self.patterns[1:] + (Nothing(),)
        for current, following in zip(self.patterns, following_patterns):
            original = current
            current = _maybe_unwrap_capture(current)
            following = _maybe_unwrap_capture(following)

            if isinstance(current, (SomeOf, SomeChunksOf)):
                if isinstance(following, (SomeOf, SomeChunksOf)):
                    following = following.delimiter

                matches = []
                while True:
                    it.checkpoint()
                    try:
                        item = next(it)
                    except StopIteration:
                        break

                    res = following.match(item, context)
                    if res is NoMatch:
                        matches.append(item)
                    else:
                        it.rewind()
                        break

                res = original.match(matches, context)
                if res is NoMatch:
                    return NoMatch
                else:
                    result.extend(res)
            else:
                try:
                    item = next(it)
                except StopIteration:
                    return NoMatch

                res = original.match(item, context)
                if res is NoMatch:
                    return NoMatch
                else:
                    result.append(res)

        return self.type(result)


def NoneOf(*args) -> Pattern:
    """Match none of the passed patterns."""
    return Not(AnyOf(*args))


def ListOf(pattern):
    """Match a list of items matching the given pattern."""
    return SequenceOf(pattern, type=list)


def TupleOf(pattern):
    """Match a variable-length tuple of items matching the given pattern."""
    return SequenceOf(pattern, type=tuple)


def DictOf(key_pattern, value_pattern):
    """Match a dictionary with keys and values matching the given patterns."""
    return MappingOf(key_pattern, value_pattern, type=dict)


def FrozenDictOf(key_pattern, value_pattern):
    """Match a frozendict with keys and values matching the given patterns."""
    return MappingOf(key_pattern, value_pattern, type=frozendict)


def pattern(obj: AnyType) -> Pattern:
    """Create a pattern from various types.

    Not that if a Coercible type is passed as argument, the constructed pattern
    won't attempt to coerce the value during matching. In order to allow type
    coercions use `Pattern.from_typehint()` factory method.

    Parameters
    ----------
    obj
        The object to create a pattern from. Can be a pattern, a type, a callable,
        a mapping, an iterable or a value.

    Examples
    --------
    >>> assert pattern(Any()) == Any()
    >>> assert pattern(int) == InstanceOf(int)
    >>>
    >>> @pattern
    ... def as_int(x, context):
    ...     return int(x)
    >>>
    >>> assert as_int.match(1, {}) == 1

    Returns
    -------
    The constructed pattern.

    """
    if obj is Ellipsis:
        return _any
    elif isinstance(obj, Pattern):
        return obj
    elif isinstance(obj, (Deferred, Resolver)):
        return Capture(obj)
    elif isinstance(obj, Mapping):
        return EqualTo(FrozenDict(obj))
    elif isinstance(obj, Sequence):
        if isinstance(obj, (str, bytes)):
            return EqualTo(obj)
        else:
            return PatternList(obj, type=type(obj))
    elif isinstance(obj, type):
        return InstanceOf(obj)
    elif get_origin(obj):
        return Pattern.from_typehint(obj, allow_coercion=False)
    elif callable(obj):
        return Custom(obj)
    else:
        return EqualTo(obj)


def match(
    pat: Pattern, value: AnyType, context: Optional[dict[str, AnyType]] = None
) -> Any:
    """Match a value against a pattern.

    Parameters
    ----------
    pat
        The pattern to match against.
    value
        The value to match.
    context
        Arbitrary mapping of values to be used while matching.

    Returns
    -------
    The matched value if the pattern matches, otherwise :obj:`NoMatch`.

    Examples
    --------
    >>> assert match(Any(), 1) == 1
    >>> assert match(1, 1) == 1
    >>> assert match(1, 2) is NoMatch
    >>> assert match(1, 1, context={"x": 1}) == 1
    >>> assert match(1, 2, context={"x": 1}) is NoMatch
    >>> assert match([1, int], [1, 2]) == [1, 2]
    >>> assert match([1, int, "a" @ InstanceOf(str)], [1, 2, "three"]) == [
    ...     1,
    ...     2,
    ...     "three",
    ... ]

    """
    if context is None:
        context = {}

    pat = pattern(pat)
    result = pat.match(value, context)
    return NoMatch if result is NoMatch else result


IsTruish = Check(bool)
IsNumber = InstanceOf(numbers.Number) & ~InstanceOf(bool)
IsString = InstanceOf(str)

As = CoercedTo
Eq = EqualTo
In = IsIn
If = Check
Some = SomeOf
