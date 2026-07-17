from __future__ import annotations

import collections.abc
from abc import abstractmethod
from typing import TYPE_CHECKING, Any
from weakref import WeakValueDictionary

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self

# AbstractMeta 是一个非常底层且关键的元类（Metaclass）。它没有选择直接继承标准库中的 abc.ABCMeta，而是通过轻量级的自定义逻辑实现了对抽象类、内存槽（__slots__）以及自定义实例化行为的精细控制
# 强制开启 __slots__（内存与性能优化）
# 在 Ibis 中，表达式（Expressions）和操作（Operations）的对象实例数量非常庞大。该元类默认会为所有子类强制自动定义 __slots__ = ()。这能阻止 Python 自动为每个实例创建内部字典 __dict__，从而大幅减少内存占用并提升属性访问速度。
# 支持轻量级抽象类（替代 abc.ABCMeta）
# abc.ABCMeta 提供了强大的虚拟子类注册功能，但在运行时进行大量的实例检查（isinstance）开销很大。AbstractMeta 自行实现了抽象方法追踪机制（检测带有 @abstractmethod 装饰器的方法）。它不允许实例化含有未实现抽象方法的类，既保留了抽象约束，又避免了运行时的性能损耗。
# 提供统一的自定义实例化入口（__create__）
# 在 Python 默认的实例化流程中，__new__ 和 __init__ 的调用绑定得比较死板（只有当 __new__ 返回该类实例时才会自动调用 __init__）。AbstractMeta 将实例化行为重定向到了自定义的 __create__ 类方法上，使得 Ibis 能够自由控制表达式对象的生命周期（例如在此阶段进行缓存拦截、直接返回已有节点或进行参数预校验）。
class AbstractMeta(type):
    """Base metaclass for many of the ibis core classes.

    Enforce the subclasses to define a `__slots__` attribute and provide a
    `__create__` classmethod to change the instantiation behavior of the class.

    Support abstract methods without extending `abc.ABCMeta`. While it provides
    a reduced feature set compared to `abc.ABCMeta` (no way to register virtual
    subclasses) but avoids expensive instance checks by enforcing explicit
    subclassing.
    """
    # 限制元类自身的属性存储
    # 作为元类（Metaclass）本身，将其 __slots__ 设置为空元组 () 可以防止元类实例（即由它创建的类对象，如 Table、Expr 等）拥有 __dict__ 或 __weakref__。这保持了元类本身在内存中的极简化。
    # 在 Python 中，你新建一个普通的类实例时，Python 会暗中为这个实例创建一个特殊的字典（__dict__）来存放它的属性。
    # 内存浪费巨大：Python 的字典（哈希表）为了保证快速查找，内部会有大量的预留空位。一个空的字典就要占用几十到上百字节。
    # 如果你在 Ibis 框架里生成了 100 万个 AST 节点（比如复杂的 SQL 树），光是这些“塑料袋（__dict__）”本身就会吃掉数百兆甚至上吉字节（GB）的内存！
    # 访问速度慢（哈希查找）：当你读取 obj.x 时，Python 必须去哈希表里“计算 x 的哈希值 -> 寻找对应的槽 -> 取出值”。这虽然快，但依然需要计算和寻址。
    # 而 AbstractMeta 元类强制让子类默认加上了 __slots__ = ()（或者子类自己指定具体的属性名，如 __slots__ = ('x', 'y')）。
    # 内存里没有任何“塑料袋”，而是只有紧挨着的两个固定内存槽位（Slots），直接放着 x 和 y。
    # 因为去掉了臃肿的哈希表字典，每个对象占用的内存瞬间缩水。
    # 既然这么好，为什么 Python 不默认开启它？因为开启 __slots__ 会失去一部分动态性：
    # 你不能再随意添加新属性了。
    # obj = SlottedClass(1, 2)
    # obj.z = 100  # 报错：AttributeError! 因为“收纳盒”没有给 z 留格子。
    __slots__ = ()

    # 在 类对象被创建时（即 Python 解释器加载代码、定义类时） 进行拦截和构建，负责注入默认槽定义并计算该类所有的抽象方法。
    # metacls：当前的元类本身（即 AbstractMeta）。
    # clsname：正在创建的子类类名（字符串）
    # bases：该子类继承的所有父类（元组）
    # dct：子类命名空间中的属性和方法字典。
    # **kwargs：其他传递给元类的关键字参数。
    def __new__(metacls, clsname, bases, dct, **kwargs):
        # enforce slot definitions
        # 检查子类的属性字典。如果子类在定义时没有显式写 __slots__，元类会自动为其补上 __slots__ = ()。这确保了 Ibis 体系下所有的子类默认都不启用 __dict__。
        dct.setdefault("__slots__", ())

        # construct the class object
        # 调用 type.__new__ 完成标准类对象的构建。
        cls = super().__new__(metacls, clsname, bases, dct, **kwargs)

        # calculate abstract methods existing in the class
        # 计算当前类定义的抽象方法
        # 扫描子类自身新定义或重写的方法中，哪些带有 @abstractmethod 装饰器（该装饰器会在方法上标记 __isabstractmethod__ = True），并收集起来。
        abstracts = {
            name
            for name, value in dct.items()
            if getattr(value, "__isabstractmethod__", False)
        }
        # 遍历所有的父类，获取父类中未实现的抽象方法集合 __abstractmethods__
        for parent in bases:
            for name in getattr(parent, "__abstractmethods__", set()):
                value = getattr(cls, name, None)
                # 如果子类（cls）中对应的该方法依然是抽象方法（即子类没有覆盖实现它，或者子类覆盖它时依然将其标记为了抽象），则将其继续保留在当前类的 abstracts 集合中。
                if getattr(value, "__isabstractmethod__", False):
                    abstracts.add(name)

        # set the abstract methods for the class
        # 将所有计算出来的抽象方法名打包成一个不可变的 frozenset 并赋值给类的 __abstractmethods__ 属性。Python 底层在实例化类时，如果发现该属性不为空，会直接抛出 TypeError 阻止实例化。
        cls.__abstractmethods__ = frozenset(abstracts)

        return cls
    # 控制 子类实例被创建时（即用户调用 MyClass(*args, **kwargs) 时） 的行为。
    # cls：当前正在被调用的子类对象。
    # *args / **kwargs：用户实例化类时传入的参数。
    def __call__(cls, *args, **kwargs):
        """Create a new instance of the class.

        The subclass may override the `__create__` classmethod to change the
        instantiation behavior. This is similar to overriding the `__new__`
        method, but without conditionally calling the `__init__` based on the
        return type.

        Parameters
        ----------
        args : tuple
            Positional arguments eventually passed to the `__init__` method.
        kwargs : dict
            Keyword arguments eventually passed to the `__init__` method.

        Returns
        -------
        The newly created instance of the class. No extra initialization

        """
        # 在标准的 Python 元类中，__call__ 默认会先后调用类的 __new__ 和 __init__ 方法。
        # 而在 AbstractMeta 中，这个默认行为被彻底改写：它直接调用并返回了 cls.__create__(*args, **kwargs) 的结果。
        # 为什么这么做？
        # 这把控制权完全交给了子类的 __create__ 类方法（Classmethod）。
        # 子类可以通过重写 __create__ 来灵活决定是返回一个新实例，还是从缓存中捞出一个已有实例（避免重复创建相同 AST 节点），或者根据传入参数的类型动态返回一个完全不同子类的实例，而不用受到 __init__ 强制初始化的羁绊。
        return cls.__create__(*args, **kwargs)

# Abstract 是 Python Ibis 核心对象体系（如表达式节点、操作节点等）的直接基类（Base Class）。它联合 AbstractMeta 一起，为整个 Ibis 框架奠定了最底层的实例化行为和内存规范。
# 核心作用有两个：
# 作为统一的基类入口，应用 AbstractMeta 的元类行为：
# 在 Python 中，元类是不会自动隐式继承的。通过让 Abstract 显式指定 metaclass=AbstractMeta，所有继承自 Abstract 的 Ibis 核心子类，都会自动继承并应用 AbstractMeta 带来的特性：
# 为整个类层级定义默认的内存与实例化契约：
# 它显式定义了子类在未重写实例化逻辑时的“默认行为”，并小心翼翼地为 Python 的底层垃圾回收/弱引用机制留出了通道。
class Abstract(metaclass=AbstractMeta):
    """Base class for many of the ibis core classes, see `AbstractMeta`."""
    # 在开启 __slots__ 的紧凑内存模式下，允许该类的实例被弱引用（Weak Reference）。
    # Abstract 显式声明 __slots__ = ("__weakref__",)。
    # 这相当于告诉 Python：“我们依然不需要肥胖的 __dict__ 属性字典，但请为我们保留一个微小的弱引用指针通道。” 这样既享受了 Slots 带来的内存暴降与速度提升，又保留了弱引用功能。
    __slots__ = ("__weakref__",)
    # 定义子类默认的实例化工厂方法
    # 它把内置的 type.__call__ 包装成一个类方法（classmethod）赋值给 __create__。
    __create__ = classmethod(type.__call__)  # type: ignore


class Immutable(Abstract):
    """Prohibit attribute assignment on the instance."""

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo) -> Self:
        return self

    def __setattr__(self, name: str, _: Any) -> None:
        raise AttributeError(
            f"Attribute {name!r} cannot be assigned to immutable instance of "
            f"type {type(self)}"
        )

# 虽然这个类名叫 Singleton（单例），但它实现的并不是传统意义上“一个类只能有一个实例”的狭义单例，而是一种基于实例化参数的享元模式（Flyweight Pattern）或对象池缓存。
# 核心作用是：根据传入的参数缓存类实例。
# 在 Ibis 框架中，如果用户多次使用相同的参数创建同一个表达式或操作节点，Singleton 可以确保只在内存中创建一次该对象，后续的调用会直接返回已经存在的同一个对象。
# 内存与性能极致优化：避免在复杂的 SQL 构建过程中，重复创建数万个内容完全相同的 AST（抽象语法树）节点。
# 极速的对象比对（$O(1)$ 复杂度）：因为相同的参数对应的是内存中同一个对象，当 Ibis 比较两个节点是否相等时，可以直接比较它们的内存地址（使用 is 运算符），这比递归对比它们内部的所有属性要快上成百上千倍。
class Singleton(Abstract):
    """Cache instances of the class based on instantiation arguments."""
    # 作为全局的对象缓存池，存放已经创建好的实例
    # 这是一个弱引用值字典（Weak Value Dictionary）。它与普通的 Python 字典 dict 最大的区别在于：它对值（Value，也就是缓存的实例对象）的引用是弱引用。
    # 自动垃圾回收（GC）：如果一个实例在外部没有任何强引用了（比如用户已经不再使用这个 Ibis 表达式了），Python 的垃圾回收器会正常将其回收。
    # 一旦被回收，WeakValueDictionary 会自动把该实例对应的键值对从字典中移除，不需要人工去清理缓存。
    __instances__: Mapping[Any, Self] = WeakValueDictionary()
    # cls：当前正在实例化的具体子类。
    # *args：传入的 positional 参数。
    # **kwargs：传入的 keyword 参数。
    @classmethod
    def __create__(cls, *args, **kwargs) -> Self:
        # 当前类本身 cls、所有的位置参数 args（本身就是元组）、以及关键字参数 kwargs（转换成只读的元组形式）组合在一起，形成了一个可哈希（Hashable）的元组 key。这个 key 唯一标识了“用这组特定参数创建该类的请求”。
        key = (cls, args, tuple(kwargs.items()))
        try:
            # 尝试从缓存中提取已有的实例：
            return cls.__instances__[key]
        except KeyError:
            instance = super().__create__(*args, **kwargs)
            cls.__instances__[key] = instance
            return instance


class Final(Abstract):
    """Prohibit subclassing."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__init_subclass__ = cls.__prohibit_inheritance__

    @classmethod
    def __prohibit_inheritance__(cls, **kwargs):
        raise TypeError(f"Cannot inherit from final class {cls}")


@collections.abc.Hashable.register
class Hashable(Abstract):
    @abstractmethod
    def __hash__(self) -> int: ...


class Comparable(Abstract):
    """Enable quick equality comparisons.

    The subclasses must implement the `__equals__` method that returns a boolean
    value indicating whether the two instances are equal. This method is called
    only if the two instances are of the same type and the result is cached for
    future comparisons.

    Since the class holds a global cache of comparison results, it is important
    to make sure that the instances are not kept alive longer than necessary.
    """

    __cache__ = {}

    @abstractmethod
    def __equals__(self, other) -> bool: ...

    def __eq__(self, other) -> bool:
        if self is other:
            return True

        # type comparison should be cheap
        if type(self) is not type(other):
            return False

        id1 = id(self)
        id2 = id(other)
        try:
            return self.__cache__[id1][id2]
        except KeyError:
            result = self.__equals__(other)
            self.__cache__.setdefault(id1, {})[id2] = result
            self.__cache__.setdefault(id2, {})[id1] = result
            return result

    __hash__ = None

    def __del__(self):
        id1 = id(self)
        for id2 in self.__cache__.pop(id1, ()):
            eqs2 = self.__cache__[id2]
            del eqs2[id1]
            if not eqs2:
                del self.__cache__[id2]


class SlottedMeta(AbstractMeta):
    def __new__(metacls, clsname, bases, dct, **kwargs):
        fields = dct.get("__fields__", dct.get("__slots__", ()))
        inherited = (getattr(base, "__fields__", ()) for base in bases)
        dct["__fields__"] = sum(inherited, ()) + fields
        return super().__new__(metacls, clsname, bases, dct, **kwargs)


class Slotted(Abstract, metaclass=SlottedMeta):
    """A lightweight alternative to `ibis.common.grounds.Annotable`.

    The class is mostly used to reduce boilerplate code.
    """

    __fields__: tuple[str, ...]

    def __init__(self, **kwargs) -> None:
        for field in self.__fields__:
            object.__setattr__(self, field, kwargs[field])

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        return all(getattr(self, n) == getattr(other, n) for n in self.__fields__)

    __hash__ = None

    def __getstate__(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__fields__}

    def __setstate__(self, state) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        fields = {k: getattr(self, k) for k in self.__fields__}
        fieldstring = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"{self.__class__.__name__}({fieldstring})"

    def __rich_repr__(self):
        for name in self.__fields__:
            yield name, getattr(self, name)


class FrozenSlotted(Slotted, Immutable, Hashable):
    """A lightweight alternative to `ibis.common.grounds.Concrete`.

    This class is used to create immutable dataclasses with slots and a precomputed
    hash value for quicker dictionary lookups.
    """

    __slots__ = ("__precomputed_hash__",)
    __fields__ = ()
    __precomputed_hash__: int

    def __init__(self, **kwargs) -> None:
        values = []
        for field in self.__fields__:
            values.append(value := kwargs[field])
            object.__setattr__(self, field, value)
        hashvalue = hash((self.__class__, tuple(values)))
        object.__setattr__(self, "__precomputed_hash__", hashvalue)

    def __setstate__(self, state):
        for name, value in state.items():
            object.__setattr__(self, name, value)
        hashvalue = hash((self.__class__, tuple(state.values())))
        object.__setattr__(self, "__precomputed_hash__", hashvalue)

    def __hash__(self) -> int:
        return self.__precomputed_hash__
