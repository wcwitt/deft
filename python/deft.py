#!/usr/bin/env python

import ctypes as ct
import numpy as np

lib = ct.cdll.LoadLibrary('/home/wcw/codes/deft/lib/libdeft.so')

class deft(object):

    def __init__(self, nx, ny, nz, vecx, vecy, vecz):

        a1 = (ct.c_double * 3) (*vecx)
        a2 = (ct.c_double * 3) (*vecy)
        a3 = (ct.c_double * 3) (*vecz)

        lib.deft_c.argtypes = [ct.c_size_t, ct.c_size_t, ct.c_size_t, \
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
        lib.deft_c.restype = ct.c_void_p

        lib.equals_c.argtypes = [ct.c_void_p, ct.c_double]
        lib.equals_c.restype = ct.c_void_p

        lib.at_c.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int]
        lib.at_c.restype = ct.c_double

        lib.copy_data_from_c.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
        lib.copy_data_from_c.restype = ct.c_void_p

        lib.integrate_c.argtypes = [ct.c_void_p]
        lib.integrate_c.restype = ct.c_double

        lib.compute_gradient_x_c.argtypes = [ct.c_void_p]
        lib.compute_gradient_x_c.restype = ct.c_void_p

        lib.compute_gradient_y_c.argtypes = [ct.c_void_p]
        lib.compute_gradient_y_c.restype = ct.c_void_p

        lib.compute_gradient_z_c.argtypes = [ct.c_void_p]
        lib.compute_gradient_z_c.restype = ct.c_void_p

        lib.compute_gradient_squared_c.argtypes = [ct.c_void_p]
        lib.compute_gradient_squared_c.restype = ct.c_void_p

        lib.compute_laplacian_c.argtypes = [ct.c_void_p]
        lib.compute_laplacian_c.restype = ct.c_void_p

        lib.interpolate_c.argtypes = [ct.c_size_t, ct.c_size_t, ct.c_size_t]
        lib.interpolate_c.restype = ct.c_void_p

        lib.sum_over_lattice_c.argtypes = [ct.c_void_p, ct.c_size_t, \
                        ct.POINTER(ct.c_double), ct.CFUNCTYPE(ct.c_double,ct.c_double)]
        lib.sum_over_lattice_c.restype = ct.c_void_p

        self.obj = lib.deft_c(nx, ny, nz, a1, a2, a3)

    def equals(self, val):
        lib.equals_c(self.obj, val)

    def at(self, i, j, k):
        return lib.at_c(self.obj, i, j, k)

    def copy_data_from(self, data):
        lib.copy_data_from_c(self.obj, data.ctypes.data_as(ct.POINTER(ct.c_double)))

    def integrate(self):
        return lib.integrate_c(self.obj)

    def compute_gradient_x(self):
        lib.compute_gradient_x_c(self.obj)

    def compute_gradient_y(self):
        lib.compute_gradient_y_c(self.obj)

    def compute_gradient_z(self):
        lib.compute_gradient_z_c(self.obj)

    def compute_gradient_squared(self):
        lib.compute_gradient_squared_c(self.obj)

    def compute_laplacian(self):
        lib.compute_laplacian_c(self.obj)

    def interpolate(self, new_x, new_y, new_z):
        return lib.interpolate_c(self.obj, new_x, new_y, new_z)

    def sum_over_lattice(self, num, loc, func):
        lib.sum_over_lattice_c(self.obj, num, loc.ctypes.data_as(ct.POINTER(ct.c_double)), func)

def fourier_interpolate(grd, new_x, new_y, new_z, ax, ay, az):

    # ensure the input arrays have the expected alignment
    grd = np.require(grd, dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    ax  = np.require(ax,  dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    ay  = np.require(ay,  dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    az  = np.require(az,  dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])

    # use deft to interpolate
    grd_deft = deft(grd.shape[0], grd.shape[1], grd.shape[2], ax, ay, az)
    out_deft = deft(new_x, new_y, new_z, ax, ay, az)
    grd_deft.copy_data_from(grd)
    out_deft.obj = grd_deft.interpolate(new_x, new_y, new_z)

    # return the result as a numpy array
    out = np.require(np.zeros([new_x,new_y,new_z], order='F'), dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    for k in range(new_z):
        for j in range(new_y):
            for i in range(new_x):
                out[i,j,k] = out_deft.at(i,j,k)
    return out
    
def compute_gradient(grd, direc, ax, ay, az):

    # ensure the input arrays have the expected alignment
    grd = np.require(grd, dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    ax  = np.require(ax,  dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    ay  = np.require(ay,  dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    az  = np.require(az,  dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])

    # use deft to compute gradient
    grd_deft = deft(grd.shape[0], grd.shape[1], grd.shape[2], ax, ay, az)
    grd_deft.copy_data_from(grd)
    if direc=='x':
        grd_deft.compute_gradient_x()
    elif direc=='y':
        grd_deft.compute_gradient_y()
    elif direc=='z':
        grd_deft.compute_gradient_z()

    # return the result as a numpy array
    out = np.require(np.zeros(grd.shape, order='F'), dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    for k in range(out.shape[2]):
        for j in range(out.shape[1]):
            for i in range(out.shape[0]):
                out[i,j,k] = grd_deft.at(i,j,k)
    return out

def compute_laplacian(grd, ax, ay, az):

    # ensure the numpy arrays have the expected alignment
    grd = np.require(grd, dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    ax  = np.require(ax,  dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    ay  = np.require(ay,  dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    az  = np.require(az,  dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])

    # use deft to compute gradient
    grd_deft = deft(grd.shape[0], grd.shape[1], grd.shape[2], ax, ay, az)
    grd_deft.copy_data_from(grd)
    grd_deft.compute_laplacian()

    # return the result as a numpy array
    out = np.require(np.zeros(grd.shape, order='F'), dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    for k in range(out.shape[2]):
        for j in range(out.shape[1]):
            for i in range(out.shape[0]):
                out[i,j,k] = grd_deft.at(i,j,k)
    return out


def sum_over_lattice(grd_pts, loc, ax, ay, az, func):

    # ensure the numpy arrays have the expected alignment
    loc = np.require(loc, dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    ax  = np.require(ax, dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    ay  = np.require(ay, dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    az  = np.require(az, dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    out = np.require(np.zeros(grd_pts, order='F'), dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    
    # use deft to perform the sum
    out_deft = deft(grd_pts[0], grd_pts[1], grd_pts[2], ax, ay, az)
    callback_type = ct.CFUNCTYPE(ct.c_double, ct.c_double)
    callback_func = callback_type(func)
    out_deft.sum_over_lattice(loc.shape[0], loc, callback_func)
    
    # return the result as a numpy array
    for k in range(out.shape[2]):
        for j in range(out.shape[1]):
            for i in range(out.shape[0]):
                out[i,j,k] = out_deft.at(i,j,k)
    return out
