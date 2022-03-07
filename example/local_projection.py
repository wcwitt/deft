import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, spherical_jn, jv, jn_zeros, comb
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.optimize import brentq

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../build/'))
import pydeft as deft

# --------------------------------------------------------
# This script bla bla
# --------------------------------------------------------

# ------------------------------------ Math Helper Functions -------------------------------------

def real_sph_harm(l, m, x, y, z):
  """
  Generates real spherical harmonics from complex ones (from SciPy).
  ----------------------------------------------------------------------
  l,m (floats)        : spherical harmonics coefficients.
  x,y,z (Numpy arays) : cartesian coordinates with shape (nx,ny,nz).
  """
  # convert cartesian to spherical polar coordinates 
  r = np.sqrt(x*x+y*y+z*z)
  theta = np.where(r>1e-12, np.arctan2(y,x), 0)
  theta += (theta<0)*2*np.pi
  z_r = np.ones(r.shape)
  np.divide(z, r, where=(r>1e-12), out=z_r)
  phi = np.arccos(z_r)

  # calculate real spherical harmonics from complex ones
  if m<0:
    return np.sqrt(2)*(-1)**m * np.imag(sph_harm(np.abs(m),l,theta,phi))
  elif m==0:
    return np.real(sph_harm(m,l,theta,phi))
  else:
    return np.sqrt(2)*(-1)**m * np.real(sph_harm(m,l,theta,phi))

def spherical_jn_zero(l, n):
  """
  Computes the n'th root of the l'th order spherical bessel function. This method
  uses the observation that root of j_{n+0.5} lies between the roots of j_n and j_{n+1}.
  """
  a, b = jn_zeros(l, n)[-1], jn_zeros(l+1, n)[-1]
  root = brentq(lambda x: spherical_jn(l, x), a, b)
  return root

def get_k_js(l, k_max, r_max):
  """
  Computes a set of spherical bessel function roots to be used in the 
  'get_coefficients' routine.
  """
  roots = []
  n = 1 # indexes the n-th root
  root = spherical_jn_zero(l, n)
  while r_max*k_max - root > 0: 
    roots.append(root)
    n += 1
    root = spherical_jn_zero(l, n)
  return np.array(roots).astype(np.float64)/r_max

def smoothstep(x, x_min=0, x_max=1, N=1):
  """
  Smoothstep function, f(x), that interpolates from 1 to 0 between some x_min 
  and x_max, with smoothness up to the N'th derivative. 
  i.e. f(x < x_min) = 1 and f(x > x_max) = 0
  """
  x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
  result = 0
  for n in range(0, N + 1):
    result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n
  return 1- result *x ** (N + 1)
    
# ----------------------------------- DEFT Helper Functions --------------------------------------
  
def get_coefficients(box_vecs, grid, l_max, k_max, r_max, data):  
  data_ft = deft.fourier_transform(data)
  box = deft.Box(box_vecs)
  kx = deft.wave_vectors_x(data_ft.shape(), box)
  ky = deft.wave_vectors_y(data_ft.shape(), box)
  kz = deft.wave_vectors_z(data_ft.shape(), box)
  k = deft.wave_numbers(data_ft.shape(), box)
      
  print('Largest wavevector k = {:.3f}'.format(np.max(k)))
  for threshold in np.linspace(0.2*np.max(k), 0.8*np.max(k), 4):
      print('Max data value for k > {:.3f} : {:.5g}'.format(threshold, np.max(np.abs(data_ft[k[...]>threshold]))))

  # getting k_js
  kjs, num_coeffs = [], 0
  for l in range(0, l_max+1):
      kjs.append(get_k_js(l, k_max, r_max))
      num_coeffs += (2*l+1) * kjs[-1].size
  print('\nNumber of coefficients to compute: {}'.format(num_coeffs))
  # calculating coefficients
  coeffs = np.zeros([grid[0], grid[1], grid[2], num_coeffs+1])
  
  # processing inputs for integral
  k_ref = np.linspace(0, k[...].max(), int(k[...].max()*100))
  r = np.linspace(0, r_max, int(100*r_max))
  ks_ref = np.full((len(r),) + k_ref.shape, k_ref).transpose(1,0)
  rs = np.full(k_ref.shape + (len(r),), r)
  work_ft = deft.Complex3D(data_ft.shape())
  i = 0
  for l in range(0, l_max+1):
      for k_j in kjs[l]:
          I_ref = simps(rs*rs*spherical_jn(l, k_j*rs)*spherical_jn(l, ks_ref*rs)*smoothstep(rs, 0, r_max, 9), rs)
          I = interp1d(k_ref, I_ref)(k[...]) 
          for m in range(-l, l+1):
              sh = real_sph_harm(l, m, kx[...], ky[...], kz[...])
              work_ft[...] = 2**(5/2) * np.pi / r_max**(3/2) / spherical_jn(l+1, k_j*r_max) * (1j)**l * I * sh * data_ft[...]
              coeffs[:,:,:,i] = deft.inverse_fourier_transform(work_ft, grid)[...]
              i += 1
              if i%10 == 0:
                  print('{}/{} coefficients calculated'.format(i, num_coeffs))
  return kjs, coeffs
			
def basis_function(k, l, m, x, y, z, r_max):
  # used for reconstructing the functions from the projected coefficients
  r = np.sqrt(x*x + y*y + z*z)
  return np.where(r < r_max, 2**(1/2)/r_max**(3/2)/spherical_jn(l+1, k*r_max) * spherical_jn(l,k*r) * real_sph_harm(l,m,x,y,z), 0.0)

def reconstruct(coeffs, box_vecs, grid, kjs, r_max, data, point): 
  l_max = len(kjs)-1
  
  # fractional coordinates
  xf, yf, zf = np.meshgrid(np.arange(grid[0])/grid[0], np.arange(grid[1])/grid[1], np.arange(grid[2])/grid[2], indexing='ij')
  # cartesian coordinates
  x = box_vecs[0,0]*xf + box_vecs[1,0]*yf + box_vecs[2,0]*zf
  y = box_vecs[0,1]*xf + box_vecs[1,1]*yf + box_vecs[2,1]*zf
  z = box_vecs[0,2]*xf + box_vecs[1,2]*yf + box_vecs[2,2]*zf

  arr = np.zeros(data.shape()); i = 0
  for l in range(0, l_max+1):
      for k_j in kjs[l]:
          for m in range(-l, l+1):
              arr += coeffs[point[0],point[1], point[2],i]*basis_function(k_j, l, m, x-x[point[0],point[1],point[2]], \
              y-y[point[0],point[1],point[2]], z-z[point[0],point[1],point[2]], r_max)
              i += 1
  return arr
                
# ------------------------------------------ Example --------------------------------------------
             
grid_size = 20
grid = np.array([grid_size, grid_size, grid_size])

length = 2.5
box_vecs = length * np.eye(3)

# generate test data
def f_tilde(kx,ky,kz):
    l0 = 0; l1 =  1; l2 = 2; l3 =  3
    m0 = 0; m1 = -1; m2 = 0; m3 = -2
    w0 = 5; w1 =  3; w2 = 2; w3 =  3
    k = np.sqrt(kx*kx+ky*ky+kz*kz)
    Y00 = 1/np.sqrt(4*np.pi)
    return((-1j*k)/(2.0*w0))**l0*np.exp(-k*k/(4.0*w0))*real_sph_harm(l0,m0,kx,ky,kz)/Y00 \
         + ((-1j*k)/(2.0*w1))**l1*np.exp(-k*k/(4.0*w1))*real_sph_harm(l1,m1,kx,ky,kz)/Y00 \
         + ((-1j*k)/(2.0*w2))**l2*np.exp(-k*k/(4.0*w2))*real_sph_harm(l2,m2,kx,ky,kz)/Y00 \
         + ((-1j*k)/(2.0*w3))**l3*np.exp(-k*k/(4.0*w3))*real_sph_harm(l3,m3,kx,ky,kz)/Y00
data = deft.array_from_lattice_sum(grid, deft.Box(box_vecs), np.array([[0,0,0], [1, 0, 0], [1,1,1]]), f_tilde)

# Projection parameters (increase order l_max to describe larger ranges)
l_max = 3
k_max = 50  # should be larger than largest wavevector
r_max = 0.6  # real-space range to describe

print('Computing coefficients ...')
kjs, coeffs = get_coefficients(box_vecs, grid, l_max, k_max, r_max, data)
print('Coefficients computed.\n')

# point about which to reconstruct
point = [int(grid[0]/2)-2, int(grid[1]/2)+2, int(grid[2]/2)]

print('Beginning reconstruction ...')
arr = reconstruct(coeffs, box_vecs, grid, kjs, r_max, data, point)
print('Reconstruction completed.\n')

# Remaining code for generating the reconstruction plot

# fractional coordinates
xf, yf, zf = np.meshgrid(np.arange(grid[0])/grid[0], np.arange(grid[1])/grid[1], \
                         np.arange(grid[2])/grid[2], indexing='ij')
# cartesian coordinates
x = box_vecs[0,0]*xf + box_vecs[1,0]*yf + box_vecs[2,0]*zf
y = box_vecs[0,1]*xf + box_vecs[1,1]*yf + box_vecs[2,1]*zf
z = box_vecs[0,2]*xf + box_vecs[1,2]*yf + box_vecs[2,2]*zf
r = np.sqrt(x*x + y*y + z*z)

# post-processing the locally reconstructed function (correcting for the smoothstep function)
r_prime = np.sqrt((x-x[point[0],point[1],point[2]])**2 + (y-y[point[0],point[1],point[2]])**2 \
                   + (z-z[point[0],point[1],point[2]])**2)                                      
aux_arr = np.zeros(arr.shape)
aux_arr[np.abs(r_prime) < 0.7*r_max] = arr[np.abs(r_prime) < 0.7*r_max]/smoothstep(r_prime, 0, r_max, 9)[np.abs(r_prime) < 0.7*r_max]
arr = aux_arr

print('Generating plot ...')
fig, ax = plt.subplots(5, 1, sharex=True)

r_100 = [r[i,0,0] for i in range(r.shape[0])]
ax[0].plot(r_100, [data[i,point[1],point[2]] for i in range(data.shape()[0])], '--b')
ax[0].plot(r_100, [arr[i,point[1],point[2]] for i in range(data.shape()[0])], 'rx')
ax[0].set_ylabel('[100]')

r_010 = [r[0,i,0] for i in range(r.shape[0])]
ax[1].plot(r_010, [data[point[0],i,point[2]] for i in range(data.shape()[1])], '--b')
ax[1].plot(r_010, [arr[point[0],i,point[2]] for i in range(data.shape()[1])], 'rx')
ax[1].set_ylabel('[010]')

r_001 = [r[0,0,i] for i in range(r.shape[0])]
ax[2].plot(r_001, [data[point[0],point[1],i] for i in range(data.shape()[2])], '--b')
ax[2].plot(r_001, [arr[point[0],point[1],i] for i in range(data.shape()[2])], 'rx')
ax[2].set_ylabel('[001]')

r_110 = [r[i,i,0] for i in range(r.shape[0])]
ax[3].plot(r_110, [data[i,i,point[2]] for i in range(data.shape()[1])], '--b')
ax[3].plot(r_110, [arr[i,i,point[2]] for i in range(data.shape()[1])], 'rx')
ax[3].set_ylabel('[110]')

r_111 = [r[i,i,i] for i in range(r.shape[0])]
ax[4].plot(r_111, [data[i,i,i] for i in range(data.shape()[2])], '--b')
ax[4].plot(r_111, [arr[i,i,i] for i in range(data.shape()[2])], 'rx')
ax[4].set_ylabel('[111]')

for i in range(5):
  ax[i].grid()
fig.legend(['Original Function', 'Local Reconstruction'], loc='lower center', ncol=2)
plt.show()    
    