from typing import Type, Union, Tuple
from . import _ir
import inspect, types, numpy

DType = Type[Union[bool, int, float]]
ArrayLike = Union[bool, int, float, numpy.ndarray, 'Array']

class Array:

  @staticmethod
  def cast(value):
    if isinstance(value, Array):
      return value
    elif isinstance(value, numpy.ndarray):
      return Constant(value)
    elif isinstance(value, (bool, int, float)):
      return Constant(numpy.ndarray(value))
    else:
      raise ValueError

  def __init__(self, shape: Tuple[int, ...], dtype: DType) -> None:
    self.shape = shape
    self.dtype = dtype

  def __add__(self, other: ArrayLike) -> 'Array':
    try:
      other = Array.cast(other)
    except ValueError:
      return NotImplemented
    return Add(self, other)

  def __mul__(self, other: ArrayLike) -> 'Array':
    try:
      other = Array.cast(other)
    except ValueError:
      return NotImplemented
    return Mul(self, other)

  def __neg__(self) -> 'Array':
    return Negate(self, other)

  def sum(self, axis: int) -> 'Array':
    return Sum(self, axis)

  def derivative(self, target: 'Argument') -> 'Array':
    return Derivative(self, target)

# Helper functions for automatically converting arguments for some `IRArray`,
# an implementation of `_ir.Array`, to arguments for an implementation of
# `Array`. The `cast` method casts an argument to the expected type, e.g.
# the `_ArrayHelper` can cast lists and numpy arrays to `Array`. The `dummy`
# method creates a dummy argument, used to compute the shape of the wrapper
# using the corresponding `IRArray`. The lower method lowers arguments, e.g.
# the `_ArrayHelper` lowers an `Array` to `_ir.Array`.

class _ArrayHelper:

  @staticmethod
  def cast(value: ArrayLike) -> Array:
    return Array.cast(value)

  @staticmethod
  def dummy(value: Array) -> _ir.Array:
    return _ir.Dummy(shape=value.shape, dtype=value.dtype)

  @staticmethod
  def lower(value: Array, kwargs) -> _ir.Array:
    return value.lower(**kwargs)

class _AxisIndexHelper:

  @staticmethod
  def cast(value: int) -> int:
    return int(value)

  @staticmethod
  def dummy(value: int) -> int:
    return value

  @staticmethod
  def lower(value: int, kwargs) -> int:
    return value + 1

class _LengthHelper:

  @staticmethod
  def cast(value: int) -> int:
    return int(value)

  @staticmethod
  def dummy(value: int) -> int:
    return value

  @staticmethod
  def lower(value: int, kwargs) -> int:
    return value

_helper_map = {_ir.ArrayArg: _ArrayHelper,
               _ir.AxisIndexArg: _AxisIndexHelper,
               _ir.LengthArg: _LengthHelper}

# Wrap `IRArray` an implementation of `_ir.Array` as an `Array`. The `IRArray` should
# have supported parameter annotations (see `_helper_map`).

def _wrap(IRArray) -> Type[Array]:

  sig = inspect.signature(IRArray)

  # Collect the helpers for each parameter of `IRArray` based on the annotation.
  helpers = {param.name: _helper_map[param.annotation] for param in sig.parameters.values()}

  def apply_helpers(meth: str, args, kwargs, *methargs):
    ba = sig.bind(*args, **kwargs)
    for name in tuple(ba.arguments):
      ba.arguments[name] = getattr(helpers[name], meth)(ba.arguments[name], *methargs)
    return ba.args, ba.kwargs

  def __init__(*args, **kwargs) -> None:
    self, *args = args

    # Cast arguments to the correct types.
    self.__args, self.__kwargs = apply_helpers('cast', args, kwargs)

    # Get the shape and dtype from the `IRArray` instantiated with dummy
    # arguments.
    dummyargs, dummykwargs = apply_helpers('dummy', self.__args, self.__kwargs)
    dummy = IRArray(*dummyargs, **dummykwargs)

    Array.__init__(self, shape=dummy.shape, dtype=dummy.dtype)

  # TODO: __init__.__signature__

  def lower(self, **lowerargs) -> _ir.Array:
    # Lower the arguments and instantiate `IRArray`.
    args, kwargs = apply_helpers('lower', self.__args, self.__kwargs, lowerargs)
    return IRArray(*args, **kwargs)

  return types.new_class(IRArray.__name__, (Array,), exec_body=lambda ns: ns.update(__init__=__init__, lower=lower))

class Argument(Array):

  def __init__(self, name: str, shape: Tuple[int, ...], dtype: DType) -> None:
    self.name = name
    super().__init__(shape=shape, dtype=dtype)

  def lower(self, **lowerargs) -> _ir.Array:
    return _ir.InsertAxis(_ir.Argument(self.name, shape=self.shape, dtype=self.dtype), 0, 1)

class Derivative(Array):

  def __init__(self, arg: Array, target: Argument) -> None:
    self.arg = arg
    self.target = target
    super().__init__(shape=arg.shape+target.shape, dtype=arg.dtype)

  def lower(self, **lowerargs) -> _ir.Array:
    target = _ir.Argument(self.target.name, self.target.shape, self.target.dtype)
    return self.arg.lower(**lowerargs).derivative(target)

class Constant(Array):

  def __init__(self, value: numpy.ndarray) -> None:
    self.value = value
    super().__init__(shape=value.shape, dtype=value.dtype)

  def lower(self, **kwargs) -> _ir.Array:
    return _ir.InsertAxis(_ir.Constant(self.value), 0, 1)

InsertAxis = _wrap(_ir.InsertAxis)
Sum = _wrap(_ir.Sum)
Add = _wrap(_ir.Add)
Mul = _wrap(_ir.Mul)
Neg = _wrap(_ir.Neg)
Cos = _wrap(_ir.Cos)
Sin = _wrap(_ir.Sin)

from unittest import TestCase

class Test(TestCase):

  def test_sum(self):
    value = Sum(numpy.array([[1, 2], [3, 4]]), 0)
    self.assertEqual(value.lower().eval().tolist(), [[4, 6]])

  def test_cos(self):
    value = Cos(numpy.array([1,2,3]))
    self.assertEqual(value.lower().eval().tolist(), numpy.cos([[1,2,3]]).tolist())

  def test_arg(self):
    arg = Argument('test', shape=(2,), dtype=int)
    self.assertEqual(arg.lower().eval(test=numpy.array([1, 2])).tolist(),
                     [[1, 2]])

  def test_derivative(self):
    arg = Argument('test', shape=(2,), dtype=int)
    self.assertEqual(Sin(arg).derivative(arg).lower().eval(test=numpy.array([1, 2])).tolist(),
                     numpy.cos([[[1, 2], [1, 2]]]).tolist())
