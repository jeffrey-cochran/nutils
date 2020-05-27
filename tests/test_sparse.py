import unittest, numpy, contextlib
from nutils import sparse


@contextlib.contextmanager
def chunksize(n):
  chunksize = sparse.chunksize
  try:
    sparse.chunksize = n
    yield
  finally:
    sparse.chunksize = chunksize


class vector(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.data = numpy.array(
      [((4,), 10),
       ((4,), 20),
       ((3,),  1),
       ((2,), 30),
       ((1,), 40),
       ((2,), 50),
       ((3,), -1),
       ((0,),  0),
       ((0,), 60)], dtype=sparse.dtype([5], int))
    self.full = numpy.array(
      [ 60, 40, 80,  0, 30 ])

  def test_issparse(self):
    self.assertTrue(sparse.issparse(self.data))
    self.assertFalse(sparse.issparse(numpy.array([10.])))

  def test_issparsedtype(self):
    self.assertTrue(sparse.issparsedtype(self.data.dtype))
    self.assertFalse(sparse.issparsedtype(numpy.dtype('int8')))

  def test_ndim(self):
    self.assertEqual(sparse.ndim(self.data), 1)

  def test_shape(self):
    self.assertEqual(sparse.shape(self.data), (5,))

  def test_indices(self):
    i0, = sparse.indices(self.data)
    self.assertEqual(i0.tolist(), [4, 4, 3, 2, 1, 2, 3, 0, 0])

  def test_values(self):
    v = sparse.values(self.data)
    self.assertEqual(v.tolist(), [10, 20, 1, 30, 40, 50, -1, 0, 60])

  def test_extract(self):
    indices, values, shape = sparse.extract(self.data)
    self.assertEqual(indices[0].tolist(), sparse.indices(self.data)[0].tolist())
    self.assertEqual(values.tolist(), sparse.values(self.data).tolist())
    self.assertEqual(shape, sparse.shape(self.data))

  def test_dedup(self):
    for inplace in False, True:
      with self.subTest(inplace=inplace), chunksize(self.data.itemsize * 3):
        dedup = sparse.dedup(self.data, inplace=inplace)
        (self.assertIs if inplace else self.assertIsNot)(dedup, self.data)
        self.assertEqual(dedup.tolist(),
          [((0,),60), ((1,),40), ((2,),80), ((3,),0), ((4,),30)])

  def test_prune(self):
    for inplace in False, True:
      with self.subTest(inplace=inplace), chunksize(self.data.itemsize * 3):
        prune = sparse.prune(self.data, inplace=inplace)
        (self.assertIs if inplace else self.assertIsNot)(prune, self.data)
        self.assertEqual(prune.tolist(),
          [((4,),10), ((4,),20), ((3,),1), ((2,),30), ((1,),40), ((2,),50), ((3,),-1), ((0,),60)])

  def test_block(self):
    A = self.data
    B = C = numpy.array([
      ((1,), 10)], dtype=sparse.dtype([3]))
    for a, b, c in numpy.ndindex(2,2,2):
      with self.subTest(A=a, B=b, C=c):
        datas = [a and A, b and B, c and C]
        if a and b and c:
          retval = sparse.block(datas)
          self.assertEqual(sparse.shape(retval), (11,))
          self.assertEqual(retval.tolist(),
            [((4,),10), ((4,),20), ((3,),1), ((2,),30), ((1,),40), ((2,),50), ((3,),-1), ((0,),0), ((0,),60), ((6,),10), ((9,),10)])
        else:
          with self.assertRaises(Exception):
            sparse.blocks(datas)

  def test_toarray(self):
    array = sparse.toarray(self.data)
    self.assertEqual(array.tolist(), self.full.tolist())

  def test_fromarray(self):
    data = sparse.fromarray(self.full)
    self.assertEqual(data.tolist(),
      [((0,),60),((1,),40),((2,),80),((3,),0),((4,),30)])

  def test_add_int(self):
    other = numpy.array([
      ((1,), -40),
      ((2,),  70)], dtype=self.data.dtype)
    retval = sparse.add([self.data, other])
    self.assertEqual(retval.dtype, self.data.dtype)
    self.assertEqual(retval.tolist(),
      [((4,),10), ((4,),20), ((3,),1), ((2,),30), ((1,),40), ((2,),50), ((3,),-1), ((0,),0), ((0,),60), ((1,),-40), ((2,),70)])

  def test_add_float(self):
    other = numpy.array([
      ((1,), -40),
      ((2,),  .5)], dtype=sparse.dtype((5,), float))
    retval = sparse.add([self.data, other])
    self.assertEqual(retval.dtype, other.dtype)
    self.assertEqual(retval.tolist(),
      [((4,),10), ((4,),20), ((3,),1), ((2,),30), ((1,),40), ((2,),50), ((3,),-1), ((0,),0), ((0,),60), ((1,),-40), ((2,),.5)])

class matrix(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.data = numpy.array([
      ((2,4), 10),
      ((3,4), 20),
      ((2,3),  1),
      ((1,2), 30),
      ((0,1), 40),
      ((1,2), 50),
      ((2,3), -1),
      ((3,0),  0),
      ((2,0), 60)], dtype=sparse.dtype([4,5], int))
    self.full = numpy.array(
      [[  0, 40,  0,  0,  0 ],
       [  0,  0, 80,  0,  0 ],
       [ 60,  0,  0,  0, 10 ],
       [  0,  0,  0,  0, 20 ]])

  def test_issparse(self):
    self.assertTrue(sparse.issparse(self.data))
    self.assertFalse(sparse.issparse(numpy.array([[10.]])))

  def test_issparsedtype(self):
    self.assertTrue(sparse.issparsedtype(self.data.dtype))
    self.assertFalse(sparse.issparsedtype(numpy.dtype('int8')))

  def test_ndim(self):
    self.assertEqual(sparse.ndim(self.data), 2)

  def test_shape(self):
    self.assertEqual(sparse.shape(self.data), (4,5))

  def test_indices(self):
    i0, i1 = sparse.indices(self.data)
    self.assertEqual(i0.tolist(), [2, 3, 2, 1, 0, 1, 2, 3, 2])
    self.assertEqual(i1.tolist(), [4, 4, 3, 2, 1, 2, 3, 0, 0])

  def test_values(self):
    v = sparse.values(self.data)
    self.assertEqual(v.tolist(), [10, 20, 1, 30, 40, 50, -1, 0, 60])

  def test_extract(self):
    indices, values, shape = sparse.extract(self.data)
    self.assertEqual(indices[0].tolist(), sparse.indices(self.data)[0].tolist())
    self.assertEqual(indices[1].tolist(), sparse.indices(self.data)[1].tolist())
    self.assertEqual(values.tolist(), sparse.values(self.data).tolist())
    self.assertEqual(shape, sparse.shape(self.data))

  def test_dedup(self):
    for inplace in False, True:
      with self.subTest(inplace=inplace), chunksize(self.data.itemsize * 3):
        dedup = sparse.dedup(self.data, inplace=inplace)
        (self.assertIs if inplace else self.assertIsNot)(dedup, self.data)
        self.assertEqual(dedup.tolist(),
          [((0,1),40), ((1,2),80), ((2,0),60), ((2,3),0), ((2,4),10), ((3,0),0), ((3,4),20)])

  def test_prune(self):
    for inplace in False, True:
      with self.subTest(inplace=inplace), chunksize(self.data.itemsize * 3):
        prune = sparse.prune(self.data, inplace=inplace)
        (self.assertIs if inplace else self.assertIsNot)(prune, self.data)
        self.assertEqual(prune.tolist(),
          [((2,4),10), ((3,4),20), ((2,3),1), ((1,2),30), ((0,1),40), ((1,2),50), ((2,3),-1), ((2,0),60)])

  def test_block(self):
    A = self.data
    B = numpy.array([((1,0), 10)], dtype=sparse.dtype([4,2]))
    C = E = numpy.array([((0,2), 20)], dtype=sparse.dtype([1,5]))
    D = F = numpy.array([((0,1), 30)], dtype=sparse.dtype([1,2]))
    for a, b, c, d, e, f in numpy.ndindex(2,2,2,2,2,2):
      with self.subTest(A=a, B=b, C=c, D=d, E=e, F=f):
        datas = [[a and A, b and B], [c and C, d and D], [e and E, f and F]]
        if (a or c or e) and (b or d or f) and (a or b) and (c or d) and (e or f):
          retval = sparse.block(datas)
          self.assertEqual(sparse.shape(retval), (6,7))
          self.assertEqual(retval.tolist(),
            ([((2,4), 10), ((3,4), 20), ((2,3), 1), ((1,2), 30), ((0,1), 40), ((1,2), 50), ((2,3), -1), ((3,0), 0), ((2,0), 60)] if a else [])
          + ([((1,5), 10)] if b else [])
          + ([((4,2), 20)] if c else [])
          + ([((4,6), 30)] if d else [])
          + ([((5,2), 20)] if e else [])
          + ([((5,6), 30)] if f else []))
        else:
          with self.assertRaises(Exception):
            sparse.blocks(datas)

  def test_toarray(self):
    array = sparse.toarray(self.data)
    self.assertEqual(array.tolist(), self.full.tolist())

  def test_fromarray(self):
    data = sparse.fromarray(self.full)
    self.assertEqual(data.tolist(),
      [((0,0),0),((0,1),40),((0,2),0),((0,3),0),((0,4),0),
       ((1,0),0),((1,1),0),((1,2),80),((1,3),0),((1,4),0),
       ((2,0),60),((2,1),0),((2,2),0),((2,3),0),((2,4),10),
       ((3,0),0),((3,1),0),((3,2),0),((3,3),0),((3,4),20)])

  def test_add_int(self):
    other = numpy.array([
      ((0, 1), -40),
      ((0, 2),  70)], dtype=self.data.dtype)
    retval = sparse.add([self.data, other])
    self.assertEqual(retval.dtype, self.data.dtype)
    self.assertEqual(retval.tolist(),
      [((2,4),10), ((3,4),20), ((2,3),1), ((1,2),30), ((0,1),40), ((1,2),50), ((2,3),-1), ((3,0),0), ((2,0),60), ((0,1),-40), ((0,2),70)])

  def test_add_float(self):
    other = numpy.array([
      ((0, 1), -40),
      ((0, 2),  .5)], dtype=sparse.dtype((4,5), float))
    retval = sparse.add([self.data, other])
    self.assertEqual(retval.dtype, other.dtype)
    self.assertEqual(retval.tolist(),
      [((2,4),10), ((3,4),20), ((2,3),1), ((1,2),30), ((0,1),40), ((1,2),50), ((2,3),-1), ((3,0),0), ((2,0),60), ((0,1),-40), ((0,2),.5)])
