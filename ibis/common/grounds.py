from __future__ import annotations

import contextlib
from copy import copy
from typing import (
    Any,
    ClassVar,
    Union,
    get_origin,
)

from typing_extensions import Self, dataclass_transform

from ibis.common.annotations import (
    Annotation,
    Argument,
    Attribute,
    Signature,
)
from ibis.common.bases import (  # noqa: F401
    Abstract,
    AbstractMeta,
    Comparable,
    Final,
    Hashable,
    Immutable,
    Singleton,
)
from ibis.common.collections import FrozenDict  # noqa: TC001
from ibis.common.patterns import Pattern
from ibis.common.typing import evaluate_annotations

# Ibis 表达式架构中最核心的元类（Metaclass）之一。
# 将 Python 类的类型注解（Type Annotations）转化为一套自动化的验证与签名系统。
# 作用是把普通的 Python 类变成一个拥有严格参数校验能力的“工厂”。
# 自动签名生成：将类属性的类型注解（如 arg: Value）自动提取，合并父类签名，生成一个完整的 __signature__。
# 强制类型约束：利用 Ibis 的 Pattern 系统，在类实例化时自动校验传入参数是否符合类型定义（如是否为 Value，是否为 int）。
# 内存优化与结构化：自动构建 __slots__ 以减少内存占用，并生成 __match_args__ 以支持 Python 的结构化模式匹配（Structural Pattern Matching）。
# 属性隔离：自动区分哪些是需要参与构造的“参数（Argument）”，哪些是内部配置的“属性（Attribute）”。
class AnnotableMeta(AbstractMeta):
    """Metaclass to turn class annotations into a validatable function signature."""

    __slots__ = ()
    # 元类的构造函数
    def __new__(metacls, clsname, bases, dct, **kwargs):
        # inherit signature from parent classes
        signatures, attributes = [], {}
        # 遍历 bases（父类），收集父类的 __attributes__（普通属性配置）和 __signature__（参数签名）。
        # 保证了子类能自动拥有父类的所有验证规则。
        for parent in bases:
            with contextlib.suppress(AttributeError):
                attributes.update(parent.__attributes__)
            with contextlib.suppress(AttributeError):
                signatures.append(parent.__signature__)

        # collection type annotations and convert them to patterns
        # 注解提取与 Pattern 转换
        module = dct.get("__module__")
        qualname = dct.get("__qualname__") or clsname
        annotations = dct.get("__annotations__", {})

        # TODO(kszucs): pass dct as localns to evaluate_annotations
        # 解析类中的 Python 类型注解（__annotations__）。

        typehints = evaluate_annotations(annotations, module, clsname)
        # 核心转换：将每一个类型注解（Type Hint）转换为一个 Pattern 对象
        for name, typehint in typehints.items():
            if get_origin(typehint) is ClassVar:
                continue
            pattern = Pattern.from_typehint(typehint)
            if name in dct:
                dct[name] = Argument(pattern, default=dct[name], typehint=typehint)
            else:
                dct[name] = Argument(pattern, typehint=typehint)

        # collect the newly defined annotations
        # 遍历类定义体（dct），将成员分为三类
        # Pattern / Argument：标记为构造参数，加入 arguments 字典，并自动添加到 __slots__。
        # Attribute：标记为元数据属性（如 shape, dtype），存入 attributes 字典。
        # 其他（方法、常量）：保留在 namespace 中，作为类的常规成员。
        slots = list(dct.pop("__slots__", []))
        namespace, arguments = {}, {}
        for name, attrib in dct.items():
            if isinstance(attrib, Pattern):
                arguments[name] = Argument(attrib)
                slots.append(name)
            elif isinstance(attrib, Argument):
                arguments[name] = attrib
                slots.append(name)
            elif isinstance(attrib, Attribute):
                attributes[name] = attrib
                slots.append(name)
            else:
                namespace[name] = attrib

        # merge the annotations with the parent annotations
        # 调用 Signature.merge 将所有参数整合成一个标准的 Python inspect.Signature 对象，存储为 __signature__。这使得 Ibis 节点可以像函数一样被 signature.bind() 检查。
        signature = Signature.merge(*signatures, **arguments)
        argnames = tuple(signature.parameters.keys())

        namespace.update(
            __module__=module,
            __qualname__=qualname,
            __argnames__=argnames,
            __attributes__=attributes,
            __match_args__=argnames,
            __signature__=signature,
            __slots__=tuple(slots),
        )
        return super().__new__(metacls, clsname, bases, namespace, **kwargs)

    def __or__(self, other):
        # required to support `dt.Numeric | dt.Floating` annotation for python<3.10
        return Union[self, other]

# Annotable 为继承它的所有类（通常是 Ibis 的 AST 表达式节点）提供了一套自动化的类型校验与生命周期管理机制：
# 参数自动校验：在对象创建时，自动检查传入的参数是否符合 AnnotableMeta 生成的类型签名。
# 属性安全更新：即便在对象创建后，修改其属性（__setattr__）时也会触发类型校验，确保对象始终处于合法状态。
# 声明式结构处理：通过标准化的 __args__ 和 __repr__ 实现，使得复杂的嵌套 AST 节点可以像普通数据类（dataclass）一样被轻松序列化、调试和比较。
@dataclass_transform()
class Annotable(Abstract, metaclass=AnnotableMeta):
    """Base class for objects with custom validation rules."""
    # 定义了类的构造参数签名（包含参数名、类型模式、默认值）。
    __signature__: ClassVar[Signature]
    """Signature of the class, containing the Argument annotations."""
    # 存储了非构造参数的元数据字段（如 dtype, shape），用于初始化默认值或验证。
    __attributes__: ClassVar[FrozenDict[str, Annotation]]
    """Mapping of the Attribute annotations."""
    # 所有核心构造参数名称的元组，用于快速遍历。
    __argnames__: ClassVar[tuple[str, ...]]
    """Names of the arguments."""
    # Python 3.10+ 模式匹配支持的参数列表，允许直接使用 case MyNode(arg1=x): 语法。
    __match_args__: ClassVar[tuple[str, ...]]
    """Names of the arguments to be used for pattern matching."""
    # 标准实例化入口
    @classmethod
    def __create__(cls, *args: Any, **kwargs: Any) -> Self:
        # construct the instance by passing only validated keyword arguments
        kwargs = cls.__signature__.validate(cls, args, kwargs)
        return super().__create__(**kwargs)

    @classmethod
    def __recreate__(cls, kwargs: Any) -> Self:
        # bypass signature binding by requiring keyword arguments only
        kwargs = cls.__signature__.validate_nobind(cls, kwargs)
        return super().__create__(**kwargs)

    def __init__(self, **kwargs: Any) -> None:
        # set the already validated arguments
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)
        # initialize the remaining attributes
        for name, field in self.__attributes__.items():
            if field.has_default():
                object.__setattr__(self, name, field.get_default(name, self))

    def __setattr__(self, name, value) -> None:
        # first try to look up the argument then the attribute
        if param := self.__signature__.parameters.get(name):
            value = param.annotation.validate(name, value, self)
        # then try to look up the attribute
        elif annot := self.__attributes__.get(name):
            value = annot.validate(name, value, self)
        return super().__setattr__(name, value)

    def __repr__(self) -> str:
        args = (f"{n}={getattr(self, n)!r}" for n in self.__argnames__)
        argstring = ", ".join(args)
        return f"{self.__class__.__name__}({argstring})"

    def __eq__(self, other) -> bool:
        # compare types
        if type(self) is not type(other):
            return NotImplemented
        # compare arguments
        if self.__args__ != other.__args__:
            return False
        # compare attributes
        for name in self.__attributes__:
            if getattr(self, name, None) != getattr(other, name, None):
                return False
        return True

    __hash__ = None

    @property
    def __args__(self) -> tuple[Any, ...]:
        return tuple(getattr(self, name) for name in self.__argnames__)

    def copy(self, **overrides: Any) -> Annotable:
        """Return a copy of this object with the given overrides.

        Parameters
        ----------
        overrides
            Argument override values

        Returns
        -------
        Annotable
            New instance of the copied object

        """
        this = copy(self)
        for name, value in overrides.items():
            setattr(this, name, value)
        return this

# Concrete 是一个集大成者。它通过多重继承，将之前讨论过的 Immutable（防篡改）、Comparable（高性能对比）和 Annotable（自动校验）融合在一起，形成了 Ibis 中最底层的、完全不可变的表达式节点基类
# Concrete 类定义了 Ibis 中“具体节点（Concrete Node）”的行为准则。它的核心目标是性能极致化与数据一致性：
# 哈希预计算（Precomputed Hash）：由于对象不可变，它在创建时一次性计算好哈希值，后续作为 dict 的 Key 或 set 的元素时，查询时间复杂度为 $O(1)$。
# 强制不可变契约：继承 Immutable，严禁任何属性修改。
# 高效序列化：通过 __reduce__ 自定义 Pickling 行为，仅存储构造参数，实现轻量级跨进程传输。
# 作为高性能缓存键：它是 Ibis 表达式缓存系统中最理想的 Key 类型。
class Concrete(Immutable, Comparable, Annotable):
    """Opinionated base class for immutable data classes."""
    # 显式定义实例内存布局。除了继承自父类的属性外，仅额外存储 __args__（参数元组）和 __precomputed_hash__（哈希缓存），极大地降低了每个节点的内存占用。
    __slots__ = ("__args__", "__precomputed_hash__")
    # 构造实例并完成“冷启动”计算
    def __init__(self, **kwargs: Any) -> None:
        # collect and set the arguments in a single pass
        args = []
        # 参数绑定：根据 __argnames__ 遍历传入的参数，使用 object.__setattr__ 绕过校验直接赋值。
        for name in self.__argnames__:
            value = kwargs[name]
            args.append(value)
            object.__setattr__(self, name, value)

        # precompute the hash value since the instance is immutable
        # 哈希预计算：将所有参数打包为 tuple，与类类型 self.__class__ 组合生成 hashvalue 并存储，确保后续 hash() 调用是常数级开销。
        args = tuple(args)
        hashvalue = hash((self.__class__, args))
        object.__setattr__(self, "__args__", args)
        object.__setattr__(self, "__precomputed_hash__", hashvalue)

        # initialize the remaining attributes
        # 默认值处理：初始化未显式传入但定义了默认值的属性。
        for name, field in self.__attributes__.items():
            if field.has_default():
                object.__setattr__(self, name, field.get_default(name, self))
    # Python Pickle 协议方法，决定对象如何被序列化。
    def __reduce__(self):
        # assuming immutability and idempotency of the __init__ method, we can
        # reconstruct the instance from the arguments without additional attributes
        state = dict(zip(self.__argnames__, self.__args__))
        return (self.__recreate__, (state,))

    def __hash__(self) -> int:
        return self.__precomputed_hash__

    # Comparable 基类要求的接口实现
    def __equals__(self, other) -> bool:
        return hash(self) == hash(other) and self.__args__ == other.__args__

    # 暴露内部存储的参数数据。
    # 对外提供只读接口，方便编译器或优化器遍历节点的所有依赖项。
    @property
    def args(self):
        return self.__args__

    @property
    def argnames(self) -> tuple[str, ...]:
        return self.__argnames__

    # 基于当前实例“浅克隆”并修改参数，生成新实例。
    def copy(self, **overrides) -> Self:
        kwargs = dict(zip(self.__argnames__, self.__args__))
        if unknown_args := overrides.keys() - kwargs.keys():
            raise AttributeError(f"Unexpected arguments: {unknown_args}")
        kwargs.update(overrides)
        return self.__recreate__(kwargs)
