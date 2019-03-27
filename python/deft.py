#!/usr/bin/env python

import ctypes as ct
import numpy as np

lib = ct.cdll.LoadLibrary('/home/wcw/codes/deft/lib/libdeft.so')

class deft(object):

    def __init__(self, nx, ny, nz, vecx, vecy, vecz):

        a1 = (ct.c_double * 3) (*vecx)
        a2 = (ct.c_double * 3) (*vecy)
        a3 = (ct.c_double * 3) (*vecz)

        lib.deft_c.argtypes = [ct.c_int, ct.c_int, ct.c_int, \
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
        lib.deft_c.restype = ct.c_void_p

        lib.equals_c.argtypes = [ct.c_void_p, ct.c_double]
        lib.equals_c.restype = ct.c_void_p

        lib.at_c.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int]
        lib.at_c.restype = ct.c_double

        lib.integrate_c.argtypes = [ct.c_void_p]
        lib.integrate_c.restype = ct.c_double

        lib.interpolate_c.argtypes = [ct.c_size_t, ct.c_size_t, ct.c_size_t]
        lib.interpolate_c.restype = ct.c_void_p

        lib.compute_periodic_superposition_c.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_double), ct.CFUNCTYPE(ct.c_double,ct.c_double)]
        lib.compute_periodic_superposition_c.restype = ct.c_void_p

        self.obj = lib.deft_c(nx, ny, nz, a1, a2, a3)

    def equals(self, val):
        lib.equals_c(self.obj, val)

    def at(self, i, j, k):
        return lib.at_c(self.obj, i, j, k)

    def copy_data_from(self, data):
        return lib.copy_data_from_c(self.obj, data.ctypes.data_as(ct.POctypes.INTER(ct.c_double)))

    def integrate(self):
        return lib.integrate_c(self.obj)

    def interpolate(self, new_x, new_y, new_z):
        return lib.interpolate_c(self.obj, new_x, new_y, new_z)

    def compute_periodic_superposition(self, num, loc, func):
        return lib.compute_periodic_superposition_c(self.obj, num, loc.ctypes.data_as(ct.POINTER(ct.c_double)), func)

def fourier_interpolate(grd, new_x, new_y, new_z, ax, ay, az):

    grd_deft = deft(grd.shape[0], grd.shape[1], grd.shape[2], ax, ay, az)
    grd_deft.copyDataFrom(grd)

    new_x_c = ct.c_size_t(new_x); new_y_c = ct.c_size_t(new_y); new_z_c = ct.c_size_t(new_z)
    arr_deft = deft(new_x, new_y, new_z, ax, ay, az)
    arr_deft.obj = grd_deft.interpolate(new_x_c, new_y_c, new_z_c)
    arr = np.require(np.zeros([new_x,new_y,new_z]), dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    for k in range(new_x):
        for j in range(new_y):
            for i in range(new_z):
                arr[i,j,k] = arr_deft.at(i,j,k)
    return arr
    
def compute_periodic_superposition(grd_pts, loc, ax, ay, az, func):

    grd_deft = deft(grd_pts[0], grd_pts[1], grd_pts[2], ax, ay, az)
    callback_type = ct.CFUNCTYPE(ct.c_double, ct.c_double)
    callback_func = callback_type(func)
    loc_aligned = np.require(loc, dtype='float64', requirements=['F_CONTIGUOUS', 'ALIGNED'])
    grd_deft.compute_periodic_superposition(loc.shape[0], loc_aligned, callback_func)
    grd = np.zeros(grd_pts)
    for k in range(grd.shape[0]):
        for j in range(grd.shape[1]):
            for i in range(grd.shape[2]):
                grd[i,j,k] = grd_deft.at(i,j,k)
    return grd
