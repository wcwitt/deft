import numpy as np
from scipy.special import sph_harm

def get_box_geometry(vectors):
    """expects box vectors in rows of 'vectors'"""

    # get box lengths, angles, and volume
    lengths = np.linalg.norm(vectors, axis=1)
    angles = np.empty(3)
    angles[0] = np.arccos(
            vectors[1,:].dot(vectors[2,:])/(lengths[1]*lengths[2]))
    angles[1] = np.arccos(
            vectors[0,:].dot(vectors[2,:])/(lengths[0]*lengths[2]))
    angles[2] = np.arccos(
            vectors[0,:].dot(vectors[1,:])/(lengths[0]*lengths[1]))
    volume = np.linalg.det(vectors)
    # get reciprocal lattice vectors (rows of 'recip_vectors') and lengths
    recip_vectors = 2.0*np.pi*np.linalg.inv(vectors).T
    recip_lengths = np.linalg.norm(recip_vectors, axis=1)
    return lengths, angles, volume, recip_vectors, recip_lengths

def get_function_on_grid(function, shape, vectors, r0=None):
    """ (1) function must take 3d numpy arrays as input
        (2) minimum image convention brings each component to within (0.5,0.5)
        (3) expects box vectors in rows of 'vectors'
    """
    if r0 is None:
        r0 = np.zeros(3)
    else:
        r0 = np.linalg.inv(vectors.T).dot(r0) # convert to scaled coords
    # compute possible elements of dr
    x = np.arange(shape[0], dtype='float')/shape[0] - r0[0]
    y = np.arange(shape[1], dtype='float')/shape[1] - r0[1]
    z = np.arange(shape[2], dtype='float')/shape[2] - r0[2]
    # bring components within [0.5,0.5]
    x = x - np.rint(x)
    y = y - np.rint(y)
    z = z - np.rint(z)
    # create grids for vectorized calls
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    # convert to cartesian coordinates
    x = vectors[0,0]*xx + vectors[1,0]*yy + vectors[2,0]*zz
    y = vectors[0,1]*xx + vectors[1,1]*yy + vectors[2,1]*zz
    z = vectors[0,2]*xx + vectors[1,2]*yy + vectors[2,2]*zz
    # evaluate function
    return function(x,y,z)

def real_sph_harm(l, m, x, y, z):

    r = np.sqrt(x*x+y*y+z*z)
    # compute azimuthal angle, domain of [0,2*pi)
    theta = np.arctan2(y,x)
    theta += (theta<0)*2*np.pi
    # compute polar angle, setting phi=0 for r->0
    phi = np.zeros(np.array(r).shape) # np.array() enables scalar x,y,z
    np.divide(z, r, out=phi, where=(r>1e-12))
    phi = np.arccos(phi)
    # return real spherical harmonic
    if m<0:
        return np.sqrt(2)*(-1)**m*np.imag(sph_harm(np.abs(m),l,theta,phi))
    elif m==0:
        return np.real(sph_harm(m,l,theta,phi))
    else:
        return np.sqrt(2)*(-1)**m*np.real(sph_harm(m,l,theta,phi))
