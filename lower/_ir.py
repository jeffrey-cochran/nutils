from typing import Type, Union, Tuple
from typing_extensions import Annotated
import numpy

ArrayArg = Annotated['Array', 'array']
AxisIndexArg = Annotated[int, 'axis_index']
LengthArg = Annotated[int, 'length']

DType = Type[Union[bool, int, float]]

class Array:

  def __init__(self, args: Tuple['Array', ...], shape: Tuple[int, ...], dtype: DType) -> None:
    self.__args = args
    self.shape = shape
    self.dtype = dtype

  def asciitree(self, prefix=''):
    return f'{prefix}{type(self).__name__}\n' + ''.join(arg.asciitree(prefix+'  ') for arg in self.__args)

  def eval(self, **evalargs):
    return self.evalf(*(arg.eval(**evalargs) for arg in self.__args))

class Dummy(Array):

  def __init__(self, shape: Tuple[int, ...], dtype: DType) -> None:
    super().__init__(args=(), shape=shape, dtype=dtype)

  def evalf(self):
    raise ValueError

  def derivative(self, var):
    raise ValueError

class Argument(Array):

  def __init__(self, name: str, shape: Tuple[int, ...], dtype: DType) -> None:
    self.name = name
    super().__init__(args=(), shape=shape, dtype=dtype)

  def eval(self, **evalargs):
    return evalargs[self.name]

  def derivative(self, var):
    return Constant((numpy.ones if var == self else numpy.zeros)(self.shape+var.shape))

  def __eq__(self, other):
    return type(self) == type(other) and self.name == other.name and self.shape == other.shape and self.dtype == other.dtype

class Constant(Array):

  def __init__(self, value: numpy.ndarray) -> None:
    self.value = value
    super().__init__(args=(), shape=value.shape, dtype=value.dtype)

  def evalf(self):
    return self.value

  def derivative(self, var):
    return Constant(numpy.zeros(self.shape+var.shape))

class InsertAxis(Array):

  def __init__(self, arg: ArrayArg, axis: AxisIndexArg, length: LengthArg) -> None:
    self.arg = arg
    self.axis = axis
    self.length = length
    super().__init__(args=(arg,), shape=arg.shape[:axis]+(length,)+arg.shape[axis+1:], dtype=arg.dtype)

  def evalf(self, arg):
    arg = arg[(slice(None),)*self.axis+(None,)]
    if self.length != 1:
      arg = numpy.repeat(arg, self.length, self.axis)
    return arg

  def derivative(self, var):
    return InsertAxis(self.arg.derivative(var), self.axis, self.length)

class Sum(Array):

  def __init__(self, arg: ArrayArg, axis: AxisIndexArg) -> None:
    self.arg = arg
    self.axis = axis
    super().__init__(args=(arg,), shape=arg.shape[:axis]+arg.shape[axis+1:], dtype=arg.dtype)

  def evalf(self, arg):
    return arg.sum(self.axis)

  def derivative(self, var):
    return Sum(self.arg.derivative(var), self.axis)

class Neg(Array):

  def __init__(self, arg: ArrayArg) -> None:
    super().__init__(args=(arg,), shape=arg.shape, dtype=arg.dtype)

  def evalf(self, arg):
    return -arg

  def derivative(self, var):
    return Neg(self.arg.derivative(var))

class Add(Array):

  def __init__(self, array1: ArrayArg, array2: ArrayArg) -> None:
    assert array1.shape == array2.shape
    self.array1 = array1
    self.array2 = array2
    super().__init__(args=(array1, array2), shape=array1.shape, dtype=array1.dtype)

  def evalf(self, array1, array2):
    return array1 + array2

  def derivative(self, var):
    return Add(array1.derivative(var), array2.derivative(var))

class Mul(Array):

  def __init__(self, array1: ArrayArg, array2: ArrayArg) -> None:
    assert array1.shape == array2.shape
    self.array1 = array1
    self.array2 = array2
    super().__init__(args=(array1, array2), shape=array1.shape, dtype=array1.dtype)

  def evalf(self, array1, array2):
    return array1 * array2

  def derivative(self, var):
    return Add(Mul(array1.derivative(var), _append_axes(array2, var.shape)),
               Mul(_append_axes(array1, var.shape), array2.derivative(var)))

class Sin(Array):

  def __init__(self, arg: ArrayArg) -> None:
    self.arg = arg
    super().__init__(args=(arg,), shape=arg.shape, dtype=float)

  def evalf(self, arg):
    return numpy.sin(arg)

  def derivative(self, var):
    return Mul(_append_axes(Cos(self.arg), var.shape), self.arg.derivative(var))

class Cos(Array):

  def __init__(self, arg: ArrayArg) -> None:
    self.arg = arg
    super().__init__(args=(arg,), shape=arg.shape, dtype=float)

  def evalf(self, arg):
    return numpy.cos(arg)

  def derivative(self, var):
    return Mul(_append_axes(Neg(Sin(self.arg)), var.shape), self.arg.derivative(var))

def _append_axes(array: Array, shape: Tuple[int, ...]) -> Array:
  for n in shape:
    array = InsertAxis(array, len(array.shape), n)
  return array
