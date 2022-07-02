import numpy as np
import os
import scipy.interpolate
import sys
import unittest

sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '../build/'))
import pydeft as deft

import tools_for_tests as tools

class TestLatticeSum(unittest.TestCase):

    def test_cardinal_b_spline_values(self):

        m = 11                 # grid points in [0,1)
        for n in range(2,31):  # spline order
            spl = np.zeros(m*n)
            for i in range(m):
                array = deft.cardinal_b_spline_values(i/m,n)
                for j in range(n):
                    spl[i+j*m] = array[j]
            x = np.linspace(0,n,m*n,endpoint=False)
            bspl = scipy.interpolate.BSpline.basis_element(range(0,n+1))
            self.assertTrue(np.allclose(spl, bspl(x)))

    def test_cardinal_b_spline_derivatives(self):

        m = 11                 # grid points in [0,1)
        for n in range(3,31):  # spline order
            spl_deriv = np.zeros(m*n)
            for i in range(m):
                array = deft.cardinal_b_spline_derivatives(i/m,n)
                for j in range(n):
                    spl_deriv[i+j*m] = array[j]
            x = np.linspace(0,n,m*n,endpoint=False)
            bspl = scipy.interpolate.BSpline.basis_element(range(0,n+1))
            self.assertTrue(np.allclose(spl_deriv, bspl.derivative()(x)))

    def test_exponential_spline_b(self):

        order = 20
        m = 3  # accuracy degrades for m>3
        N = 9
        x = np.linspace(0,8,20,endpoint=False)
        f = np.exp(2*np.pi*1j*m/N*x)
        s = np.zeros(x.size, dtype=complex)
        for i in range(x.size):
            for k in range(-50,50):
                if x[i]-k<=0 or x[i]-k>=order:
                    continue
                M = deft.cardinal_b_spline_values(x[i]-k-np.floor(x[i]-k),order)
                s[i] += M[int(np.floor(x[i]-k))]*np.exp(2*np.pi*1j*m/N*k)
        s *= deft.exponential_spline_b(m,N,order)
        self.assertTrue(np.allclose(f, s))

    def test_structure_factor_spline(self):

        shape = (35,36,37)
        grd = deft.Double3D(shape)
        box = deft.Box([[4.9,0.1,0.2], [-0.2,5.0,0.3], [0.3,-0.1,5.1]])
        xyz = np.array([[0,0,0], [2,0.1,0.2], [0.3,1,2]])
        str_fac = deft.structure_factor(shape, box, xyz)
        str_fac_spline = deft.structure_factor_spline(shape, box, xyz, 20)
        # spline approx only highly accurate for low-freq wavevectors
        t = 10
        self.assertTrue(
              np.allclose(str_fac[:t,:t,:t], str_fac_spline[:t,:t,:t])
            * np.allclose(str_fac[:t,-t:,:t], str_fac_spline[:t,-t:,:t])
            * np.allclose(str_fac[-t:,:t,:t], str_fac_spline[-t:,:t,:t])
            * np.allclose(str_fac[-t:,-t:,:t], str_fac_spline[-t:,-t:,:t]))

    def test_array_from_lattice_sum(self):

        # skip test if multithreading is enabled
        if (os.environ.get('OMP_NUM_THREADS') is None or
                int(os.environ.get('OMP_NUM_THREADS')) > 1):
            print('\nOMP_NUM_THREADS is unset or greater than 1: '
                        'skipping test_array_from_lattice_sum.\n')
            return

        # ----- first test, spherically symmetric functions -----
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
        # test against same data generated with deft and fourier transforms
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
        self.assertTrue(np.allclose(grd[...], data))
        # repeat test with splined structure factor
        xyz = np.array([[0,0,0],
                        [2,0,0],
                        [0,1,2]])
        def f_tilde(kx,ky,kz):
            w = 3.0
            return np.exp(-(kx*kx+ky*ky+kz*kz)/(4.0*w))
        grd = deft.array_from_lattice_sum(shape, box, xyz, f_tilde, 10)
        xyz = np.array([[3,2,1]])
        def f_tilde(kx,ky,kz):
            w = 5.0
            return np.exp(-(kx*kx+ky*ky+kz*kz)/(4.0*w))
        grd += deft.array_from_lattice_sum(shape, box, xyz, f_tilde, 10)
        self.assertTrue(np.allclose(grd[...], data))

        # ----- second test, non-symmetric functions -----
        # choose grid and box, and construct deft objects
        shape = (35,35,35)
        grd = deft.Double3D(shape)
        box_vectors = 6.0*np.eye(3)
        box = deft.Box(box_vectors)
        # prepare spherical harmonics
        Y00 = 1/np.sqrt(4*np.pi)
        rsm = tools.real_sph_harm
        # superimpose functions centered at (1,2,3)
        # the functions have the form:  r^l * gaussian * Ylm/Y00
        def f(x,y,z):
            r = np.sqrt(x*x+y*y+z*z)
            def g(w,r):
                return (w/np.pi)**1.5*np.exp(-w*r*r)
            return (  r**1*g(3,r)*rsm(1,1,x,y,z)/Y00
                    + r**2*g(4,r)*rsm(2,0,x,y,z)/Y00
                    + r**3*g(3,r)*rsm(3,-1,x,y,z)/Y00 )
        data = tools.get_function_on_grid(
                f, shape, box_vectors, np.array([1,2,3]))
        # test against same data generated with deft and fourier transforms
        def f_tilde(kx,ky,kz):
            k = np.sqrt(kx*kx+ky*ky+kz*kz)
            def g_tilde(w,k):
                return np.exp(-k*k/(4*w))
            return (  (-1j*k/6)**1*g_tilde(3,k)*rsm(1,1,kx,ky,kz)/Y00
                    + (-1j*k/8)**2*g_tilde(4,k)*rsm(2,0,kx,ky,kz)/Y00
                    + (-1j*k/6)**3*g_tilde(3,k)*rsm(3,-1,kx,ky,kz)/Y00 )
        grd = deft.array_from_lattice_sum(shape, box, [[1,2,3]], f_tilde)
        self.assertTrue(np.allclose(grd[...], data))
        # repeat test with splined structure factor
        grd = deft.array_from_lattice_sum(shape, box, [[1,2,3]], f_tilde, 20)
        self.assertTrue(np.allclose(grd[...], data))

if __name__ == '__main__':
    unittest.main()
