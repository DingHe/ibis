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

# Ibis 表达式树“不可变性（Immutability）”的守护者
# Immutable 类的核心职责是强制禁止对实例属性进行任何修改。
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

# Hashable 类是一个桥接型基类。
# 通过 Python 的抽象基类（ABC）机制，正式将 Ibis 的表达式节点纳入 Python 的标准哈希协议中。
# 明确契约：强制子类通过实现 __hash__ 方法，声明该对象是“可哈希的”。
# 类型集成：通过 @collections.abc.Hashable.register，使得 Ibis 的对象在 Python 运行时可以被 isinstance(obj, collections.abc.Hashable) 正确识别。
# 这对于将表达式节点放入 set 或作为 dict 的 key 至关重要。
@collections.abc.Hashable.register
class Hashable(Abstract):
    # ... 表示“抽象方法”或“占位符”
    # 在 Python 的类型提示（Type Hinting）和存根文件（.pyi）中，... 的主要作用是告知解释器该方法尚未实现具体逻辑，或者该方法是一个接口声明。
    @abstractmethod
    def __hash__(self) -> int: ...

# 编译器和查询优化过程中，Ibis 需要频繁判断两个表达式节点是否完全一致（例如判断两棵子树是否相同以进行公共表达式消除）。
# 如果每次对比都递归遍历整棵树，开销巨大；Comparable 通过缓存计算结果，将后续相同对象的对比开销降至 $O(1)$
# 比较结果缓存：利用全局字典 __cache__ 存储已计算过的比较结果。如果 A == B 已经计算过，下次直接返回结果，避免重复执行深层逻辑。
# 强制实现契约：要求子类必须实现 __equals__ 方法，将“业务逻辑的相等性判断”与“框架级别的缓存机制”解耦。
# 内存生命周期管理：在对象销毁时自动清理缓存，防止内存泄漏。
class Comparable(Abstract):
    """Enable quick equality comparisons.

    The subclasses must implement the `__equals__` method that returns a boolean
    value indicating whether the two instances are equal. This method is called
    only if the two instances are of the same type and the result is cached for
    future comparisons.

    Since the class holds a global cache of comparison results, it is important
    to make sure that the instances are not kept alive longer than necessary.
    """
    # 类级别的全局字典，用于存储比较结果
    # 采用嵌套字典结构，key 为对象的内存 ID（id(self)），value 为另一个字典（{other_id: bool_result}）
    # 双向缓存，即存入 A 对 B 的结果时，同时记录 B 对 A 的结果
    __cache__ = {}

    @abstractmethod
    def __equals__(self, other) -> bool: ...
    # 重写了 Python 的内置相等运算符（==）。这是缓存机制的入口
    def __eq__(self, other) -> bool:
        # 如果内存地址相同，必然相等，直接返回。
        if self is other:
            return True

        # type comparison should be cheap
        # 不同类型的节点永远不相等。
        if type(self) is not type(other):
            return False
        # 缓存查找
        id1 = id(self)
        id2 = id(other)
        try:
            return self.__cache__[id1][id2]
        except KeyError:
            result = self.__equals__(other)
            self.__cache__.setdefault(id1, {})[id2] = result
            self.__cache__.setdefault(id2, {})[id1] = result
            return result
    # 显式禁止该对象被哈希化（放入 set 或作为 dict 的 key）
    __hash__ = None

    def __del__(self):
        id1 = id(self)
        for id2 in self.__cache__.pop(id1, ()):
            eqs2 = self.__cache__[id2]
            del eqs2[id1]
            if not eqs2:
                del self.__cache__[id2]

# SlottedMeta 元类是用来自动化管理与继承类字段（Fields）与槽（Slots）定义的核心机制。
# 在创建类（Class Creation Phase）时，自动向上递归收集所有父类的字段/槽定义（__fields__ 或 __slots__），并将它们与当前类自身定义的字段合并，最终在类上生成一个完整的 __fields__ 元组。
# 解决继承下的属性字段收集问题：在复杂的类继承树中（如 Ibis 内部各种复杂的 IR 节点继承链），每个派生类可能都会新增或重写字段。SlottedMeta 能够确保每一个子类都能无缝感知到从所有父类继承来的全量属性字段。
# 配合内存与底层属性映射：为 Ibis 内部的对象（如 Node、Value 等结构）提供统一的元数据检索，方便属性访问校验、序列化、复制以及模式匹配等操作。
class SlottedMeta(AbstractMeta):
    # metacls：当前的元类本身（即 SlottedMeta）。
    # clsname：正在被创建的类的名字（字符串，如 "MyRelationNode"）。
    # bases：正在被创建的类的直接父类元组（tuple）。
    # dct：类的属性与方法字典（包含了类定义体中写的所有变量与函数）。
    # **kwargs：创建类时传递的其他关键字参数（如 Python 3 中的 class MyClass(metaclass=SlottedMeta, kw=value):）。
    def __new__(metacls, clsname, bases, dct, **kwargs):
        # 获取当前正在创建的类自身定义的字段/槽。
        fields = dct.get("__fields__", dct.get("__slots__", ()))
        # 获取所有直接父类中已积累的 __fields__ 字段。
        inherited = (getattr(base, "__fields__", ()) for base in bases)
        # 合并父类继承的字段与当前类定义的字段，并更新类字典。
        dct["__fields__"] = sum(inherited, ()) + fields
        return super().__new__(metacls, clsname, bases, dct, **kwargs)

# Slotted 是 Ibis 内部定义的一个轻量级底层数据类（Data Class）基类。
# 减少样板代码（Boilerplate Reduction）：它是 ibis.common.grounds.Annotable 的轻量化替代品。
# 子类只需声明字段名称（通过 __fields__ 或 __slots__），Slotted 就能自动为你生成初始化（__init__）、相等性比较（__eq__）、状态序列化/反序列化（__getstate__/__setstate__）以及可读的字符串输出（__repr__）。
# 结合 SlottedMeta 实现字段自动收集：因为它的元类是 SlottedMeta，继承 Slotted 的子类会自动获得从所有父类递归拼接而来的 __fields__ 清单。
# 基于槽的性能与规范优化：通过限制实例属性必须在 __fields__ 内，避免了常规类字典开销，便于进行高速的属性存取和对象复制。
class Slotted(Abstract, metaclass=SlottedMeta):
    """A lightweight alternative to `ibis.common.grounds.Annotable`.

    The class is mostly used to reduce boilerplate code.
    """
    # 记录当前类及其所有父类所拥有的全量属性字段名称。
    # Slotted 中的所有魔法方法（如构造函数、比较、序列化）都是完全依赖 __fields__ 中列出的字段名来驱动的。
    __fields__: tuple[str, ...]
    # 根据传入的关键字参数，自动初始化对象在 __fields__ 中定义的所有属性。
    def __init__(self, **kwargs) -> None:
        # 遍历 self.__fields__ 中的每一个字段名 field。
        # 调用底层 object.__setattr__(self, field, kwargs[field]) 将 kwargs 中对应的值绑定到实例上。
        for field in self.__fields__:
            # 使用 object.__setattr__ 可以绕过子类可能重写的 __setattr__ 限制（例如只读属性控制）。
            object.__setattr__(self, field, kwargs[field])
    # 判断当前对象与另一个对象在逻辑上是否完全相等
    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        return all(getattr(self, n) == getattr(other, n) for n in self.__fields__)
    # 显式将该类的对象标记为不可哈希（Unhashable）。
    __hash__ = None
    # 导出实例的状态字典，用于对象序列化（如 pickle 或深拷贝 deepcopy）。
    def __getstate__(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__fields__}
    # 根据传入的状态字典还原实例的属性状态。
    def __setstate__(self, state) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)
    # 生成可读性良好的对象开发者字符串表示。
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
