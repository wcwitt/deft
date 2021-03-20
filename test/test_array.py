import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '../build/'))
import pydeft as deft

import tools_for_tests as tools

class TestArray(unittest.TestCase):

    def test_shape_size_strides_unravel(self):

        # ----- Double3D -----
        # create numpy array and deft array
        shape = np.array([2,3,4], dtype='int')
        a1 = np.linspace(0.0, 20.0, shape.prod()).reshape(shape)
        arr1 = deft.Double3D(shape)
        # test shape, size, and strides
        self.assertTrue(arr1.shape() == list(a1.shape))
        self.assertTrue(arr1.size(), len(a1))
        self.assertTrue(
                arr1.strides() == [int(s/a1.itemsize) for s in a1.strides])
        for i in range(a1.size):
            self.assertSequenceEqual(
                arr1.unravel_index(i), np.unravel_index(i,shape))

        # ----- Complex3D -----
        # create numpy array and deft array
        shape = np.array([2,3,4], dtype='int')
        a1 = np.linspace(0.0, 20.0, shape.prod()).reshape(shape)
        arr1 = deft.Complex3D(shape)
        # test shape, size, and strides
        self.assertTrue(arr1.shape() == list(a1.shape))
        self.assertTrue(arr1.size(), len(a1))
        self.assertTrue(
                arr1.strides() == [int(s/a1.itemsize) for s in a1.strides])
        for i in range(a1.size):
            self.assertSequenceEqual(
                arr1.unravel_index(i), np.unravel_index(i,shape))

    def test_getters_and_setters(self):

        # ----- Double3D -----
        # create two deft arrays and two numpy arrays
        shape = (2, 3, 4)
        (arr1, arr2) = (deft.Double3D(shape), deft.Double3D(shape))
        a1 = np.linspace(
                0.0, 20.0, arr1.size(), dtype=arr1[...].dtype).reshape(shape)
        a2 = 1.0 + np.indices(shape, dtype=arr2[...].dtype).sum(axis=0)
        # test single index __getitem__ and __setitem__
        for u in range(np.prod(shape)):
            arr1[u] = a1.flatten()[u]
            arr2[u] = arr1[u]
        self.assertTrue(np.allclose(arr1[...], arr2[...]))
        # test multi-index __getitem__ and __setitem__
        for u in range(shape[0]):
            for v in range(shape[1]):
                for w in range(shape[2]):
                    arr2[u,v,w] = a2[u,v,w]
                    arr1[u,v,w] = arr2[u,v,w]
        self.assertTrue(np.allclose(arr1[...], arr2[...]))
        # test spliced __getitem__ and __setitem__
        arr1[1,:,2] = a1[1,:,2]
        self.assertFalse(np.allclose(arr1[...], arr2[...]))
        self.assertTrue(np.allclose(arr1[1,:,2], a1[1,:,2]))

        # ----- Complex3D -----
        # create two deft arrays and two numpy arrays
        shape = (2, 3, 4)
        (arr1, arr2) = (deft.Complex3D(shape), deft.Complex3D(shape))
        a1 = np.linspace(
                0.0, 20.0, arr1.size(), dtype=arr1[...].dtype).reshape(shape)
        a2 = 1.0 + np.indices(shape, dtype=arr2[...].dtype).sum(axis=0)
        # test single index __getitem__ and __setitem__
        for u in range(np.prod(shape)):
            arr1[u] = a1.flatten()[u]
            arr2[u] = arr1[u]
        self.assertTrue(np.allclose(arr1[...], arr2[...]))
        # test multi-index __getitem__ and __setitem__
        for u in range(shape[0]):
            for v in range(shape[1]):
                for w in range(shape[2]):
                    arr2[u,v,w] = a2[u,v,w]
                    arr1[u,v,w] = arr2[u,v,w]
        self.assertTrue(np.allclose(arr1[...], arr2[...]))
        # test spliced __getitem__ and __setitem__
        arr1[1,:,2] = a1[1,:,2]
        self.assertFalse(np.allclose(arr1[...], arr2[...]))
        self.assertTrue(np.allclose(arr1[1,:,2], a1[1,:,2]))

    def test_arithmetic_assignments(self):

        # ----- Double3D -----
        # create two deft arrays and two numpy arrays
        shape = (2, 3, 4)
        (arr1, arr2) = (deft.Double3D(shape), deft.Double3D(shape))
        a1 = np.linspace(
                0.0, 20.0, arr1.size(), dtype=arr1[...].dtype).reshape(shape)
        a2 = 1.0 + np.indices(shape, dtype=arr2[...].dtype).sum(axis=0)
        # test += operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr1 += 2.0
        self.assertTrue(np.allclose(arr1[...], a1+2.0))
        arr1 += arr2
        self.assertTrue(np.allclose(arr1[...], a1+2.0+a2))
        # test -= operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr1 -= 2.0
        self.assertTrue(np.allclose(arr1[...], a1-2.0))
        arr1 -= arr2
        self.assertTrue(np.allclose(arr1[...], a1-2.0-a2))
        # test *= operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr1 *= 2.0
        self.assertTrue(np.allclose(arr1[...], a1*2.0))
        arr1 *= arr2
        self.assertTrue(np.allclose(arr1[...], a1*2.0*a2))
        # test /= operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr1 /= 2.0
        self.assertTrue(np.allclose(arr1[...], a1/2.0))
        arr1 /= arr2
        self.assertTrue(np.allclose(arr1[...], a1/2.0/a2))

        # ----- Complex3D -----
        # create two deft arrays and two numpy arrays
        shape = (2, 3, 4)
        (arr1, arr2) = (deft.Complex3D(shape), deft.Complex3D(shape))
        a1 = np.linspace(
                0.0, 20.0, arr1.size(), dtype=arr1[...].dtype).reshape(shape)
        a2 = 1.0 + np.indices(shape, dtype=arr2[...].dtype).sum(axis=0)
        # test += operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr1 += 2.0
        self.assertTrue(np.allclose(arr1[...], a1+2.0))
        arr1 += arr2
        self.assertTrue(np.allclose(arr1[...], a1+2.0+a2))
        # test -= operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr1 -= 2.0
        self.assertTrue(np.allclose(arr1[...], a1-2.0))
        arr1 -= arr2
        self.assertTrue(np.allclose(arr1[...], a1-2.0-a2))
        # test *= operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr1 *= 2.0
        self.assertTrue(np.allclose(arr1[...], a1*2.0))
        arr1 *= arr2
        self.assertTrue(np.allclose(arr1[...], a1*2.0*a2))
        # test /= operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr1 /= 2.0
        self.assertTrue(np.allclose(arr1[...], a1/2.0))
        arr1 /= arr2
        self.assertTrue(np.allclose(arr1[...], a1/2.0/a2))

    def test_elementwise_math(self):

        # ----- Double3D -----
        # create numpy array and deft array
        shape = (2, 3, 4)
        arr1 = deft.Double3D(shape)
        a1 = np.linspace(
                0.0, 20.0, arr1.size(), dtype=arr1[...].dtype).reshape(shape)
        # test sqrt
        arr1[...] = a1
        arr1.compute_sqrt()
        self.assertTrue(np.allclose(arr1[...], np.sqrt(a1)))
        # test pow
        arr1[...] = a1
        arr1.compute_pow(1.0/3.0)
        self.assertTrue(np.allclose(arr1[...], a1**(1.0/3.0)))

        # ----- Complex3D -----
        # create numpy array and deft array
        shape = (2, 3, 4)
        arr1 = deft.Complex3D(shape)
        a1 = np.linspace(
                0.0, 20.0, arr1.size(), dtype=arr1[...].dtype).reshape(shape)
        # test sqrt
        arr1[...] = a1
        arr1.compute_sqrt()
        self.assertTrue(np.allclose(arr1[...], np.sqrt(a1)))
        # test pow
        arr1[...] = a1
        arr1.compute_pow(1.0/3.0)
        self.assertTrue(np.allclose(arr1[...], a1**(1.0/3.0)))

    def test_negation_addition_subtraction_multiplication_division(self):

        # ----- Double3D -----
        # create two deft arrays and two numpy arrays
        shape = (2, 3, 4)
        (arr1, arr2) = (deft.Double3D(shape), deft.Double3D(shape))
        a1 = np.linspace(
                0.0, 20.0, arr1.size(), dtype=arr1[...].dtype).reshape(shape)
        a2 = 1.0 + np.indices(shape, dtype=arr2[...].dtype).sum(axis=0)
        # test unary - operator
        arr1[...] = a1
        arr1 = -arr1
        self.assertTrue(np.allclose(arr1[...], -a1))
        # test + operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr3 = arr1 + arr2
        self.assertTrue(np.allclose(arr3[...], a1+a2))
        arr3 = arr1 + 2.0
        self.assertTrue(np.allclose(arr3[...], a1+2.0))
        arr3 = 3.0 + arr2
        self.assertTrue(np.allclose(arr3[...], 3.0+a2))
        # test - operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr3 = arr1 - arr2
        self.assertTrue(np.allclose(arr3[...], a1-a2))
        arr3 = arr1 - 2.0
        self.assertTrue(np.allclose(arr3[...], a1-2.0))
        arr3 = 3.0 - arr2
        self.assertTrue(np.allclose(arr3[...], 3.0-a2))
        # test * operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr3 = arr1 * arr2
        self.assertTrue(np.allclose(arr3[...], a1*a2))
        arr3 = arr1 * 2.0
        self.assertTrue(np.allclose(arr3[...], a1*2.0))
        arr3 = 3.0 * arr2
        self.assertTrue(np.allclose(arr3[...], 3.0*a2))
        # test / operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr3 = arr1 / arr2
        self.assertTrue(np.allclose(arr3[...], a1/a2))
        arr3 = arr1 / 2.0
        self.assertTrue(np.allclose(arr3[...], a1/2.0))
        arr3 = 3.0 / arr2
        self.assertTrue(np.allclose(arr3[...], 3.0/a2))

        # ----- Complex3D -----
        # create two deft arrays and two numpy arrays
        shape = (2, 3, 4)
        (arr1, arr2) = (deft.Complex3D(shape), deft.Complex3D(shape))
        a1 = np.linspace(
                0.0, 20.0, arr1.size(), dtype=arr1[...].dtype).reshape(shape)
        a2 = 1.0 + np.indices(shape, dtype=arr2[...].dtype).sum(axis=0)
        # test unary - operator
        arr1[...] = a1
        arr1 = -arr1
        self.assertTrue(np.allclose(arr1[...], -a1))
        # test + operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr3 = arr1 + arr2
        self.assertTrue(np.allclose(arr3[...], a1+a2))
        arr3 = arr1 + 2.0
        self.assertTrue(np.allclose(arr3[...], a1+2.0))
        arr3 = 3.0 + arr2
        self.assertTrue(np.allclose(arr3[...], 3.0+a2))
        # test - operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr3 = arr1 - arr2
        self.assertTrue(np.allclose(arr3[...], a1-a2))
        arr3 = arr1 - 2.0
        self.assertTrue(np.allclose(arr3[...], a1-2.0))
        arr3 = 3.0 - arr2
        self.assertTrue(np.allclose(arr3[...], 3.0-a2))
        # test * operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr3 = arr1 * arr2
        self.assertTrue(np.allclose(arr3[...], a1*a2))
        arr3 = arr1 * 2.0
        self.assertTrue(np.allclose(arr3[...], a1*2.0))
        arr3 = 3.0 * arr2
        self.assertTrue(np.allclose(arr3[...], 3.0*a2))
        # test / operator
        (arr1[...], arr2[...]) = (a1, a2)
        arr3 = arr1 / arr2
        self.assertTrue(np.allclose(arr3[...], a1/a2))
        arr3 = arr1 / 2.0
        self.assertTrue(np.allclose(arr3[...], a1/2.0))
        arr3 = 3.0 / arr2
        self.assertTrue(np.allclose(arr3[...], 3.0/a2))

if __name__ == '__main__':
    unittest.main()
