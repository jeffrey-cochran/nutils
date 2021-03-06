# Copyright (c) 2014 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

'''
The sample module defines the :class:`Sample` class, which represents a
collection of discrete points on a topology and is typically formed via
:func:`nutils.topology.Topology.sample`. Any function evaluation starts from
this sampling step, which drops element information and other topological
properties such as boundaries and groups, but retains point positions and
(optionally) integration weights. Evaluation is performed by subsequent calls
to :func:`Sample.integrate`, :func:`Sample.integral` or :func:`Sample.eval`.

Besides the location of points, :class:`Sample` also keeps track of point
connectivity through its :attr:`Sample.tri` and :attr:`Sample.hull`
properties, representing a (n-dimensional) triangulation of the interior and
boundary, respectively. Availability of these properties depends on the
selected sample points, and is typically used in combination with the "bezier"
set.

In addition to :class:`Sample`, the sample module defines the :class:`Integral`
class which represents postponed integration. Integrals are internally
represented as pairs of :class:`Sample` and :class:`nutils.function.Array`
objects. Evaluation proceeds via either the :func:`Integral.eval` method, or
the :func:`eval_integrals` function. The latter can also be used to evaluate
multiple integrals simultaneously, which has the advantage that it can
efficiently combine common substructures.
'''

from . import types, points, util, function, parallel, numeric, matrix, transformseq, sparse
from .pointsseq import PointsSequence
import numpy, numbers, collections.abc, os, treelog as log, abc

graphviz = os.environ.get('NUTILS_GRAPHVIZ')

def argdict(arguments):
  if len(arguments) == 1 and 'arguments' in arguments and isinstance(arguments['arguments'], collections.abc.Mapping):
    arguments = arguments['arguments']
  return types.frozendict[types.strictstr,types.frozenarray](arguments)

class Sample(types.Singleton):
  '''Collection of points on a topology.

  The :class:`Sample` class represents a collection of discrete points on a
  topology and is typically formed via :func:`nutils.topology.Topology.sample`.
  Any function evaluation starts from this sampling step, which drops element
  information and other topological properties such as boundaries and groups,
  but retains point positions and (optionally) integration weights. Evaluation
  is performed by subsequent calls to :func:`integrate`, :func:`integral` or
  :func:`eval`.

  Besides the location of points, ``Sample`` also keeps track of point
  connectivity through its :attr:`tri` and :attr:`hull` properties,
  representing a (n-dimensional) triangulation of the interior and boundary,
  respectively. Availability of these properties depends on the selected sample
  points, and is typically used in combination with the "bezier" set.
  '''

  __slots__ = 'nelems', 'transforms', 'points', 'ndims'
  __cache__ = 'allcoords'

  @staticmethod
  @types.apply_annotations
  def new(transforms:types.tuple[transformseq.stricttransforms], points:types.strict[PointsSequence], index:types.tuple[types.frozenarray[int]]=None):
    '''Create a new :class:`Sample`.

    Parameters
    ----------
    transforms : :class:`tuple` or transformation chains
        List of transformation chains leading to local coordinate systems that
        contain points.
    points : :class:`~nutils.pointsseq.PointsSequence`
        Points sequence.
    index : :class:`tuple` of integer arrays, optional
        List of indices matching ``transforms``, defining the order on which
        points show up in the evaluation. If absent the indices will be strict
        increasing.
    '''

    if index is None:
      return _DefaultIndex(transforms, points)
    else:
      return _CustomIndex(transforms, points, index)

  def __init__(self, transforms, points):
    '''
    parameters
    ----------
    transforms : :class:`tuple` or transformation chains
        List of transformation chains leading to local coordinate systems that
        contain points.
    points : :class:`~nutils.pointsseq.PointsSequence`
        Points sequence.
    '''

    assert len(transforms) >= 1
    assert all(len(t) == len(points) for t in transforms)
    self.nelems = len(transforms[0])
    self.transforms = transforms
    self.points = points
    self.ndims = transforms[0].fromdims

  def __repr__(self):
    return '{}.{}<{}D, {} elems, {} points>'.format(type(self).__module__, type(self).__qualname__, self.ndims, self.nelems, self.npoints)

  @property
  def npoints(self):
    return self.points.npoints

  @property
  def index(self):
    return tuple(map(self.getindex, range(self.nelems)))

  @abc.abstractmethod
  def getindex(self, ielem):
    '''Return the indices of `Sample.points[ielem]` in results of `Sample.eval`.'''

    raise NotImplementedError

  def _prepare_funcs(self, funcs):
    return [function.asarray(func).prepare_eval(ndims=self.ndims) for func in funcs]

  @util.positional_only
  @util.single_or_multiple
  @types.apply_annotations
  def integrate(self, funcs, arguments:argdict=...):
    '''Integrate functions.

    Args
    ----
    funcs : :class:`nutils.function.Array` object or :class:`tuple` thereof.
        The integrand(s).
    arguments : :class:`dict` (default: None)
        Optional arguments for function evaluation.
    '''

    datas = self.integrate_sparse(funcs, arguments)
    with log.iter.fraction('assembling', datas) as items:
      return [_convert(data, inplace=True) for data in items]

  @util.single_or_multiple
  @types.apply_annotations
  def integrate_sparse(self, funcs:types.tuple[function.asarray], arguments:types.frozendict[str,types.frozenarray]=None):
    '''Integrate functions into sparse data.

    Args
    ----
    funcs : :class:`nutils.function.Array` object or :class:`tuple` thereof.
        The integrand(s).
    arguments : :class:`dict` (default: None)
        Optional arguments for function evaluation.
    '''

    if arguments is None:
      arguments = {}

    # Functions may consist of several blocks, such as originating from
    # chaining. Here we make a list of all blocks consisting of triplets of
    # argument id, evaluable index, and evaluable values.

    funcs = self._prepare_funcs(funcs)
    blocks = [(ifunc, function.Tuple(ind), f.simplified.optimized_for_numpy) for ifunc, func in enumerate(funcs) for ind, f in function.blocks(func)]
    block2func, indices, values = zip(*blocks) if blocks else ([],[],[])

    log.debug('integrating {} distinct blocks'.format('+'.join(
      str(block2func.count(ifunc)) for ifunc in range(len(funcs)))))

    if graphviz:
      function.Tuple(values).graphviz(graphviz)

    # To allocate (shared) memory for all block data we evaluate indexfunc to
    # build an nblocks x nelems+1 offset array. In the first step the block
    # sizes are evaluated.

    offsets = numpy.zeros((len(blocks), self.nelems+1), dtype=numpy.uint64)
    if blocks:
      sizefunc = function.stack([f.size for ifunc, ind, f in blocks]).simplified
      for ielem, transforms in enumerate(zip(*self.transforms)):
        offsets[:,ielem+1], = sizefunc.eval(_transforms=transforms, **arguments)

    # In the second step the block sizes are accumulated to form offsets. Since
    # several blocks may belong to the same function, we post process the
    # offsets to form consecutive intervals in longer arrays. The length of
    # these arrays is captured in the nvals array.

    nvals = numpy.zeros(len(funcs), dtype=numpy.uint64)
    for iblock, ifunc in enumerate(block2func):
      v = offsets[iblock]
      v[0] = nvals[ifunc]
      numpy.cumsum(v, out=v) # in place accumulation
      if (v[1:] < v[:-1]).any():
        raise Exception('integer overflow')
      nvals[ifunc] = v[-1]

    # In a second, parallel element loop, value and index are evaluated and
    # stored in shared memory using the offsets array for location. Each
    # element has its own location so no locks are required.

    datas = [parallel.shempty(n, dtype=sparse.dtype(funcs[ifunc].shape)) for ifunc, n in enumerate(nvals)]
    valueindexfunc = function.Tuple(function.Tuple([value]+list(index)) for value, index in zip(values, indices))
    with parallel.ctxrange('integrating', self.nelems) as ielems:
      for ielem in ielems:
        points = self.points[ielem]
        for iblock, (intdata, *indices) in enumerate(valueindexfunc.eval(_transforms=tuple(t[ielem] for t in self.transforms), _points=points.coords, **arguments)):
          data = datas[block2func[iblock]][offsets[iblock,ielem]:offsets[iblock,ielem+1]].reshape(intdata.shape[1:])
          numpy.einsum('p,p...->...', points.weights, intdata, out=data['value'])
          for idim, ii in enumerate(indices):
            data['index']['i'+str(idim)] = ii.reshape([-1]+[1]*(data.ndim-1-idim))

    return datas

  def integral(self, func):
    '''Create Integral object for postponed integration.

    Args
    ----
    func : :class:`nutils.function.Array`
        Integrand.
    '''

    return Integral([(self, func)], shape=func.shape)

  @util.positional_only
  @util.single_or_multiple
  @types.apply_annotations
  def eval(self, funcs, arguments:argdict=...):
    '''Evaluate function.

    Args
    ----
    funcs : :class:`nutils.function.Array` object or :class:`tuple` thereof.
        The integrand(s).
    arguments : :class:`dict` (default: None)
        Optional arguments for function evaluation.
    '''

    funcs = self._prepare_funcs(funcs)
    retvals = [parallel.shzeros((self.npoints,)+func.shape, dtype=func.dtype) for func in funcs]
    idata = function.Tuple(function.Tuple([ifunc, function.Tuple(ind), f.simplified.optimized_for_numpy]) for ifunc, func in enumerate(funcs) for ind, f in function.blocks(func))

    if graphviz:
      idata.graphviz(graphviz)

    with parallel.ctxrange('evaluating', self.nelems) as ielems:
      for ielem in ielems:
        for ifunc, inds, data in idata.eval(_transforms=tuple(t[ielem] for t in self.transforms), _points=self.points[ielem].coords, **arguments):
          numpy.add.at(retvals[ifunc], numpy.ix_(self.getindex(ielem), *[ind for (ind,) in inds]), data)

    return retvals

  @property
  def allcoords(self):
    coords = numpy.empty([self.npoints, self.ndims])
    for ielem, points in enumerate(self.points):
      coords[self.getindex(ielem)] = points.coords
    return types.frozenarray(coords, copy=False)

  def basis(self):
    '''Basis-like function that for every point in the sample evaluates to the
    unit vector corresponding to its index.'''

    index, tail = function.TransformsIndexWithTail(self.transforms[0], function.TRANS)
    I = function.Elemwise(self.index, index, dtype=int)
    B = function.Sampled(function.ApplyTransforms(tail), expect=function.take(self.allcoords, I, axis=0))
    return function.Inflate(func=B, dofmap=I, length=self.npoints, axis=0)

  def asfunction(self, array):
    '''Convert sampled data to evaluable array.

    Using the result of :func:`Sample.eval`, create a
    :class:`nutils.function.Sampled` array that upon evaluation recovers the
    original function in the set of points matching the original sampling.

    >>> from nutils import mesh
    >>> domain, geom = mesh.rectilinear([1,2])
    >>> gauss = domain.sample('gauss', 2)
    >>> data = gauss.eval(geom)
    >>> sampled = gauss.asfunction(data)
    >>> domain.integrate(sampled, degree=2)
    array([ 1.,  2.])

    Args
    ----
    array :
        The sampled data.
    '''

    return function.matmat(self.basis(), array)

  @property
  def tri(self):
    '''Triangulation of interior.

    A two-dimensional integer array with ``ndims+1`` columns, of which every
    row defines a simplex by mapping vertices into the list of points.
    '''

    return numpy.concatenate([self.getindex(ielem).take(points.tri) for ielem, points in enumerate(self.points)])

  @property
  def hull(self):
    '''Triangulation of the exterior hull.

    A two-dimensional integer array with ``ndims`` columns, of which every row
    defines a simplex by mapping vertices into the list of points. Note that
    the hull often does contain internal element boundaries as the
    triangulations originating from separate elements are disconnected.
    '''

    return numpy.concatenate([self.getindex(ielem).take(points.hull) for ielem, points in enumerate(self.points)])

  def subset(self, mask):
    '''Reduce the number of points.

    Simple selection mechanism that returns a reduced Sample based on a
    selection mask. Points that are marked True will still be part of the new
    subset; points marked False may be dropped but this is not guaranteed. The
    point order of the original Sample is preserved.

    Args
    ----
    mask : :class:`bool` array.
        Boolean mask that selects all points that should remain. The resulting
        Sample may contain more points than this, but not less.

    Returns
    -------
    subset : :class:`Sample`
    '''

    selection = types.frozenarray([ielem for ielem in range(self.nelems) if mask[self.getindex(ielem)].any()])
    transforms = tuple(transform[selection] for transform in self.transforms)
    return Sample.new(transforms, self.points.take(selection))

strictsample = types.strict[Sample]

class _DefaultIndex(Sample):

  __slots__ = ()
  __cache__ = 'offsets'

  @property
  def offsets(self):
    return numpy.cumsum([0]+[p.npoints for p in self.points])

  def getindex(self, ielem):
    return numpy.arange(self.offsets[ielem], self.offsets[ielem+1])

  @property
  def tri(self):
    return self.points.tri

  @property
  def hull(self):
    return self.points.hull

class _CustomIndex(Sample):

  __slots__ = '_index'

  def __init__(self, transforms, points, index):
    self._index = index
    if len(index) != len(points):
      raise ValueError('expected an `index` with {} items but got {}'.format(len(points), len(index)))
    if not all(len(i) == p.npoints for i, p in zip(self._index, points)):
      raise ValueError('lengths of indices does not match number of points per element')
    super().__init__(transforms, points)

  @property
  def index(self):
    return self._index

  def getindex(self, ielem):
    return self._index[ielem]

class Integral(types.Singleton):
  '''Postponed integration.

  The :class:`Integral` class represents postponed integration. Integrals are
  internally represented as pairs of :class:`Sample` and
  :class:`nutils.function.Array` objects. Evaluation proceeds via either the
  :func:`eval` method, or the :func:`eval_integrals` function. The latter can
  also be used to evaluate multiple integrals simultaneously, which has the
  advantage that it can efficiently combine common substructures.

  Integrals support basic arithmetic such as summation, subtraction, and scalar
  multiplication and division. It also supports differentiation via the
  :func:`derivative` method. This makes Integral particularly well suited for
  use in combination with the :mod:`nutils.solver` module which provides linear
  and non-linear solvers.

  Args
  ----
  integrands : :class:`dict`
      Dictionary representing a sum of integrals, where every key-value pair
      binds together the sample set and the integrand.
  shape : :class:`tuple`
      Array dimensions of the integral.
  '''

  __slots__ = '_integrands', 'shape'
  __cache__ = 'derivative'

  @types.apply_annotations
  def __init__(self, integrands:types.frozendict[strictsample, function.simplified], shape:types.tuple[int]):
    assert all(ig.shape == shape for ig in integrands.values()), 'incompatible shapes: expected {}, got {}'.format(shape, ', '.join({str(ig.shape) for ig in integrands.values()}))
    self._integrands = {topo: func for topo, func in integrands.items() if not function.iszero(func)}
    self.shape = shape

  @property
  def ndim(self):
    return len(self.shape)

  def __repr__(self):
    return 'Integral<{}>'.format(','.join(map(str, self.shape)))

  def eval(self, **kwargs):
    '''Evaluate integral.

    Equivalent to :func:`eval_integrals` (self, ...).
    '''

    retval, = eval_integrals(self, **kwargs)
    return retval

  def derivative(self, target):
    '''Differentiate integral.

    Return an Integral in which all integrands are differentiated with respect
    to a target. This is typically used in combination with
    :class:`nutils.function.Namespace`, in which targets are denoted with a
    question mark (e.g. ``'?dofs_n'`` corresponds to target ``'dofs'``).

    Args
    ----
    target : :class:`str`
        Name of the derivative target.

    Returns
    -------
    derivative : :class:`Integral`
    '''

    argshape = self._argshape(target)
    arg = function.Argument(target, argshape)
    seen = {}
    return Integral({di: function.derivative(integrand, var=arg, seen=seen) for di, integrand in self._integrands.items()}, shape=self.shape+argshape)

  def replace(self, arguments):
    '''Return copy with arguments applied.

    Return a copy of self in which all all arguments are edited into the
    integrands. The effect is that ``self.eval(..., arguments=args)`` is
    equivalent to ``self.replace(args).eval(...)``. Note, however, that after
    the replacement it is no longer possible to take derivatives against any of
    the targets in ``arguments``.

    Args
    ----
    arguments : :class:`dict`
        Arguments for function evaluation.

    Returns
    -------
    replaced : :class:`Integral`
    '''

    return Integral({di: function.replace_arguments(integrand, arguments) for di, integrand in self._integrands.items()}, shape=self.shape)

  def contains(self, name):
    '''Test if target occurs in any of the integrands.

    Args
    ----
    name : :class:`str`
        Target name.

    Returns
    _______
    iscontained : :class:`bool`
    '''

    try:
      self._argshape(name)
    except KeyError:
      return False
    else:
      return True

  def __add__(self, other):
    if not isinstance(other, Integral):
      return NotImplemented
    assert self.shape == other.shape
    integrands = self._integrands.copy()
    for di, integrand in other._integrands.items():
      try:
        integrands[di] += integrand
      except KeyError:
        integrands[di] = integrand
    return Integral(integrands.items(), shape=self.shape)

  def __neg__(self):
    return Integral({di: -integrand for di, integrand in self._integrands.items()}, shape=self.shape)

  def __sub__(self, other):
    return self + (-other)

  def __mul__(self, other):
    if not isinstance(other, numbers.Number):
      return NotImplemented
    return Integral({di: integrand * other for di, integrand in self._integrands.items()}, shape=self.shape)

  __rmul__ = __mul__

  def __truediv__(self, other):
    if not isinstance(other, numbers.Number):
      return NotImplemented
    return self.__mul__(1/other)

  def _argshape(self, name):
    assert isinstance(name, str)
    shapes = {func.shape[:func.ndim-func._nderiv]
      for func in function.Tuple(self._integrands.values()).dependencies
        if isinstance(func, function.Argument) and func._name == name}
    if not shapes:
      raise KeyError(name)
    assert len(shapes) == 1, 'inconsistent shapes for argument {!r}'.format(name)
    shape, = shapes
    return shape

  @property
  def T(self):
    return Integral({sample: func.T for sample, func in self._integrands.items()}, shape=self.shape[::-1])

strictintegral = types.strict[Integral]

@types.apply_annotations
def eval_integrals(*integrals: types.tuple[strictintegral], **arguments:argdict):
  '''Evaluate integrals.

  Evaluate one or several postponed integrals. By evaluating them
  simultaneously, rather than using :func:`Integral.eval` on each integral
  individually, integrations will be grouped per Sample and jointly executed,
  potentially increasing efficiency.

  Args
  ----
  integrals : :class:`tuple` of integrals
      Integrals to be evaluated.
  arguments : :class:`dict` (default: None)
      Optional arguments for function evaluation.

  Returns
  -------
  results : :class:`tuple` of arrays and/or :class:`nutils.matrix.Matrix` objects.
  '''

  with log.iter.fraction('assembling', eval_integrals_sparse(*integrals, **arguments)) as retvals:
    return [_convert(retval, inplace=True) for retval in retvals]

@types.apply_annotations
def eval_integrals_sparse(*integrals: types.tuple[strictintegral], **arguments: argdict):
  '''Evaluate integrals into sparse data.

  Evaluate one or several postponed integrals. By evaluating them
  simultaneously, rather than using :func:`Integral.eval` on each integral
  individually, integrations will be grouped per Sample and jointly executed,
  potentially increasing efficiency.

  Args
  ----
  integrals : :class:`tuple` of integrals
      Integrals to be evaluated.
  arguments : :class:`dict` (default: None)
      Optional arguments for function evaluation.

  Returns
  -------
  results : :class:`tuple` of arrays and/or :class:`nutils.matrix.Matrix` objects.
  '''

  if arguments is None:
    arguments = types.frozendict({})

  retvals = [[sparse.empty(integral.shape)] for integral in integrals] # initialize with zeros to set shape and avoid empty addition
  with log.iter.fraction('topology', util.gather((di, iint) for iint, integral in enumerate(integrals) for di in integral._integrands)) as gathered:
    for sample, iints in gathered:
      for iint, retval in zip(iints, sample.integrate_sparse([integrals[iint]._integrands[sample] for iint in iints], arguments)):
        retvals[iint].append(retval)
      del retval

  return [sparse.add(retval) for retval in retvals]

def _convert(data, inplace=False):
  '''Convert a two-dimensional sparse object to an appropriate object.

  The return type is determined based on dimension: a zero-dimensional object
  becomes a scalar, a one-dimensional object a (dense) Numpy vector, a
  two-dimensional object a Nutils matrix, and any higher dimensional object a
  deduplicated and pruned sparse object.
  '''

  ndim = sparse.ndim(data)
  return sparse.toarray(data) if ndim < 2 \
    else matrix.fromsparse(data, inplace=inplace) if ndim == 2 \
    else sparse.prune(sparse.dedup(data, inplace=inplace), inplace=True)

# vim:sw=2:sts=2:et
