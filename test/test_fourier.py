import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '../build/'))
import pydeft as deft

import tools_for_tests as tools

class TestFourier(unittest.TestCase):

    def test_fourier_transform(self):

        # choose grid and construct array
        shape = (11, 13, 15)
        arr = deft.Double3D(shape)
        # generate data
        np.random.seed(0)
        data = np.random.random_sample(shape).astype('float64', 'C')
        arr[...] = data
        # compute fourier transform and compare with numpy result
        arr_ft = deft.fourier_transform(arr)
        ft = np.fft.rfftn(data) / np.prod(shape)
        self.assertTrue(np.allclose(arr_ft[...], ft))

    def test_inverse_fourier_transform(self):

        # choose grid and construct complex array
        shape = (12, 14, 16)
        arr_ft = deft.Complex3D((shape[0], shape[1], int(shape[2]/2+1)))
        # generate data
        np.random.seed(1)
        data = np.random.random_sample(shape).astype('float64', 'C')
        data_ft = np.fft.rfftn(data) / np.prod(shape)
        arr_ft[...] = data_ft
        # compute inverse fourier transform and compare with numpy result
        arr = deft.inverse_fourier_transform(arr_ft, shape)
        self.assertTrue(np.allclose(arr[...], data))

    def test_wave_vectors(self):

        # choose grid shape and box vectors
        shape = (5, 8, 11)
        box_vectors = np.array([[5.0, 0.1, -0.1],
                                 [-0.2, 5.5, 0.2],
                                 [0.3, -0.3, 6.0]])
        # get wave vector arrays with deft
        shape_ft = (shape[0], shape[1], int(shape[2]/2+1))
        ft = deft.Complex3D(shape_ft)
        box = deft.Box(box_vectors)
        wvx = deft.wave_vectors_x(ft.shape(), box)
        wvy = deft.wave_vectors_y(ft.shape(), box)
        wvz = deft.wave_vectors_z(ft.shape(), box)
        wn = deft.wave_numbers(ft.shape(), box)
        # reciprocal lattice vectors
        b = 2.0*np.pi*np.linalg.inv(box_vectors.T).T
        # k-vector indices (enforcing nyquist > 0 for even lengths)
        j0, j1 = (np.fft.fftfreq(shape[i])*shape[i] for i in range(2))
        for f in [j0, j1]: f[int(f.size/2)] = abs(f[int(f.size/2)])
        j2 = np.fft.rfftfreq(shape[2])*shape[2]
        # k-vector arrays
        kx, ky, kz, k = (np.zeros(shape_ft, order='C') for _ in range(4))
        for u in range(shape_ft[0]):
            for v in range(shape_ft[1]):
                for w in range(shape_ft[2]):
                    kvec = j0[u] * b[:,0] + j1[v] * b[:,1] + j2[w] * b[:,2]
                    kx[u,v,w], ky[u,v,w], kz[u,v,w] = kvec
                    k[u,v,w] = np.linalg.norm(kvec)
        # check equality
        self.assertTrue(np.allclose(wvx[...], kx))
        self.assertTrue(np.allclose(wvy[...], ky))
        self.assertTrue(np.allclose(wvz[...], kz))
        self.assertTrue(np.allclose(wn[...], k))

    def test_fourier_derivatives(self):

        # choose grid shape and box, and construct deft objects
        shape = (35, 37, 39)
        box_vectors = np.array([[5.0, 0.1, -0.1],
                                 [-0.2, 5.5, 0.2],
                                 [0.3, -0.3, 6.0]])
        grd = deft.Double3D(shape)
        box = deft.Box(box_vectors)
        # parameter for gaussian function
        w = 5.0
        # test gaussian on grid
        def f(x,y,z):
            return (w/np.pi)**1.5*np.exp(-w*(x*x+y*y+z*z))
        data = tools.get_function_on_grid(f, shape, box_vectors)
        grd[...] = data
        self.assertTrue(np.isclose(deft.integrate(grd,box), 1.0))
        # test gradient, x direction
        grd[...] = data
        grd = deft.gradient_x(grd, box)
        def f(x,y,z):
            return -2.0*w*x*((w/np.pi)**1.5*np.exp(-w*(x*x+y*y+z*z)))
        grad_x = tools.get_function_on_grid(f, shape, box_vectors)
        self.assertTrue(np.allclose(grd[...], grad_x))
        # test gradient, y direction
        grd[...] = data
        grd = deft.gradient_y(grd, box)
        def f(x,y,z):
            return -2.0*w*y*((w/np.pi)**1.5*np.exp(-w*(x*x+y*y+z*z)))
        grad_y = tools.get_function_on_grid(f, shape, box_vectors)
        self.assertTrue(np.allclose(grd[...], grad_y))
        # test gradient, z direction
        grd[...] = data
        grd = deft.gradient_z(grd, box)
        def f(x,y,z):
            return -2.0*w*z*((w/np.pi)**1.5*np.exp(-w*(x*x+y*y+z*z)))
        grad_z = tools.get_function_on_grid(f, shape, box_vectors)
        self.assertTrue(np.allclose(grd[...], grad_z))
        # test |\nabla f(r)|^2
        grd[...] = data
        grd = deft.grad_dot_grad(grd, box)
        g_dot_g = grad_x*grad_x + grad_y*grad_y + grad_z*grad_z
        self.assertTrue(np.allclose(grd[...], g_dot_g))
        # test laplacian
        grd[...] = data
        grd = deft.laplacian(grd, box)
        def f(x,y,z):
            rr = x*x + y*y + z*z
            return (4.0*w*w*rr-6.0*w)*((w/np.pi)**1.5*np.exp(-w*rr))
        lapl = tools.get_function_on_grid(f, shape, box_vectors)
        self.assertTrue(np.allclose(grd[...], lapl))

    def test_array_from_lattice_sum(self):

        # skip test if multithreading is enabled
        if (os.environ.get('OMP_NUM_THREADS') is None or
                int(os.environ.get('OMP_NUM_THREADS')) > 1):
            print('\nOMP_NUM_THREADS is unset or greater than 1: '
                        'skipping test_array_from_lattice_sum.\n')
            return
        # choose grid and box, and construct deft objects
        shape = (35, 35, 35)
        grd = deft.Double3D(shape)
        box_vectors = 5.0*np.eye(3)
        box = deft.Box(box_vectors)
        # superimpose gaussians with w=3.0 at (0,0,0), (2,0,0), and (0,1,2),
        # along with single gaussian with w=5.0 at (3,2,1)
        def f(x,y,z):
            w = 3.0
            return (w/np.pi)**1.5*np.exp(-w*(x*x+y*y+z*z))
        data = tools.get_function_on_grid(f, shape, box_vectors)
        data += tools.get_function_on_grid(
                f, shape, box_vectors, np.array([2.0,0.0,0.0]))
        data += tools.get_function_on_grid(
                f, shape, box_vectors, np.array([0.0,1.0,2.0]))
        def f(x,y,z):
            w = 5.0
            return (w/np.pi)**1.5*np.exp(-w*(x*x+y*y+z*z))
        data += tools.get_function_on_grid(
                f, shape, box_vectors, np.array([3.0,2.0,1.0]))
        # generate the same data with deft and fourier transforms
        xyz = np.array([[0,0,0],
                        [2,0,0],
                        [0,1,2]])
        def f_tilde(kx,ky,kz):
            w = 3.0
            return np.exp(-(kx*kx+ky*ky+kz*kz)/(4.0*w))
        grd = deft.array_from_lattice_sum(shape, box, xyz, f_tilde)
        xyz = np.array([[3,2,1]])
        def f_tilde(kx,ky,kz):
            w = 5.0
            return np.exp(-(kx*kx+ky*ky+kz*kz)/(4.0*w))
        grd += deft.array_from_lattice_sum(shape, box, xyz, f_tilde)
        # test equality
        self.assertTrue(np.allclose(grd[...], data))

if __name__ == '__main__':
    unittest.main()
