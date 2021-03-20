import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '../build/'))
import pydeft as deft

import tools_for_tests as tools

class TestBox(unittest.TestCase):

    def test_basics(self):

        vectors = np.array([[2.0, 0.1, 0.2],  # box vectors in rows
                            [0.3, 3.0, 0.4],
                            [0.5, 0.6, 4.0]])
        box = deft.Box(vectors)

        lengths, angles, volume, recip_vectors, recip_lengths = \
                tools.get_box_geometry(vectors)
        for i in range(3):
            self.assertTrue(np.allclose(box.vectors()[i], vectors[i,:]))
        self.assertTrue(np.allclose(box.lengths(), lengths))
        self.assertTrue(np.allclose(box.angles(), angles))
        self.assertTrue(np.allclose(box.volume(), volume))
        for i in range(3):
            self.assertTrue(box.recip_vectors()[i], recip_vectors[i,:])
        self.assertTrue(np.allclose(box.recip_lengths(), recip_lengths))

    def test_wave_vectors(self):

        vectors = np.array([[2.0, 0.1, 0.2],  # box vectors in rows
                            [0.3, 3.0, 0.4],
                            [0.5, 0.6, 4.0]])
        box = deft.Box(vectors)
        
        b = 2.0*np.pi*np.linalg.inv(vectors)  # recip vectors in columns
        range_0 = range(-4, 6)
        range_1 = range(-4, 6)
        range_2 = range(-4, 6)
        error = np.empty([len(range_0), len(range_1), len(range_2)])
        error_x = np.empty([len(range_0), len(range_1), len(range_2)])
        error_y = np.empty([len(range_0), len(range_1), len(range_2)])
        error_z = np.empty([len(range_0), len(range_1), len(range_2)])
        for u in range_0:
            for v in range_1:
                for w in range_2:
                    kx = u*b[0,0] + v*b[0,1] + w*b[0,2]
                    ky = u*b[1,0] + v*b[1,1] + w*b[1,2]
                    kz = u*b[2,0] + v*b[2,1] + w*b[2,2]
                    k = np.sqrt(kx*kx + ky*ky + kz*kz)
                    error[u,v,w] = k - box.wave_numbers(u,v,w)
                    error_x[u,v,w] = kx - box.wave_vectors_x(u,v,w)
                    error_y[u,v,w] = ky - box.wave_vectors_y(u,v,w)
                    error_z[u,v,w] = kz - box.wave_vectors_z(u,v,w)
        self.assertTrue(np.allclose(error, 0.0))
        self.assertTrue(np.allclose(error_x, 0.0))
        self.assertTrue(np.allclose(error_y, 0.0))
        self.assertTrue(np.allclose(error_z, 0.0))

    def test_set(self):

        vectors = np.array([[2.0, 0.1, 0.2],  # box vectors in rows
                            [0.3, 3.0, 0.4],
                            [0.5, 0.6, 4.0]])
        box = deft.Box(vectors)

        new_vectors = np.eye(3)
        box.set(new_vectors)

        lengths, angles, volume, recip_vectors, recip_lengths = \
                tools.get_box_geometry(new_vectors)
        for i in range(3):
            self.assertTrue(np.allclose(box.vectors()[i], new_vectors[i,:]))
        self.assertTrue(np.allclose(box.lengths(), lengths))
        self.assertTrue(np.allclose(box.angles(), angles))
        self.assertTrue(np.allclose(box.volume(), volume))
        for i in range(3):
            self.assertTrue(box.recip_vectors()[i], recip_vectors[i,:])
        self.assertTrue(np.allclose(box.recip_lengths(), recip_lengths))

if __name__ == '__main__':
    unittest.main()
