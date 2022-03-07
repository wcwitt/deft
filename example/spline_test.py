import numpy as np
from timeit import default_timer as timer

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../build/'))
import pydeft as deft

# Gaussian in Fourier space
def f_tilde(kx,ky,kz):
  w = 0.2
  return np.exp(-(kx*kx+ky*ky+kz*kz)/(4.0*w))

# define grid and box, and construct deft objects
shape = (50, 50, 50)
grd = deft.Double3D(shape)
box_vectors = 10*np.eye(3)
box = deft.Box(box_vectors)

print('{:^10} {:^20} {:^20}'.format('N', 'Naive', 'Spline'))

for N in np.linspace(1, 1000000, 20):
  Ni = round(N**(1/3))
  x = np.linspace(0, 10, Ni)
  x,y,z = np.meshgrid(x,x,x, indexing = 'ij')
  
  points_array = np.empty((3, Ni, Ni, Ni))
  points_array[0,:,:,:] = x
  points_array[1,:,:,:] = y
  points_array[2,:,:,:] = z
  points_array = points_array.reshape(3, Ni*Ni*Ni).T
  
  start = timer()
  grd = deft.array_from_lattice_sum(shape, box, points_array, f_tilde)
  end = timer()
  naive_time = end - start
  
  start = timer()
  grd = deft.array_from_lattice_sum(shape, box, points_array, f_tilde, 10)
  end = timer()
  spline_time = end - start
  
  print('{:^10} {:^20.4f} {:^20.4f}'.format(Ni**3, naive_time, spline_time))
