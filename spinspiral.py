from ase.io import read
import numpy as np
from ase.build import make_supercell

#THIS CODE contains functions to create the spin spiral structure.
#The idea is that one should be able to just call "construct_full"
#So in our code, we simply write:
# from spinspiral import construct_full
#then run supercell,name = construct_full(theta,phi,path,Q,magnitude,transform,magsymbols)


def cif_reader(path, transform = None):
    """Provide path to cif file"""
    primitive = read(path)
    name = primitive.get_chemical_formula(mode='metal')

    #Define transformation matrix from primitive to magnetic cell
    if transform is None:
        P = np.array([
            [2, 1, 0],
            [-1, 1, 0],
            [0, 0, 1]
        ])
    else:
        P = transform

    supercell = make_supercell(primitive, P)

    return primitive, supercell, name

def rotation_matrix(axis, theta):
    """
    Compute the rotation matrix for rotation around a given axis by theta radians.
    Uses quaternion representation for numerical stability.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)  # Normalize the axis
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    # Quaternion to rotation matrix
    R = np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
    ])
    return R

def create_spin_spiral(atoms, primitive, Q, m0, n_hat, magnetic_symbols=None):
    """
    Create a spin-spiral magnetic structure.

    Parameters:
    - atoms: ASE Atoms object (supercell)
    - primitive: ASE Atoms object (primitive cell)
    - Q: Ordering vector in reciprocal lattice units of primitive cell (2D or 3D array-like)
    - m0: Initial/reference spin vector (3D array-like)
    - n_hat: Normal vector to the spin spiral plane (3D array-like, will be normalized)
    - magnetic_symbols: List of symbols for magnetic atoms (default: ['Mn'])

    Returns:
    - atoms: The Atoms object with magnetic moments set
    """
    if magnetic_symbols is None:
        magnetic_symbols = ['Mn']
    Q     = np.asarray(Q, dtype=float)
    m0    = np.asarray(m0, dtype=float)
    n_hat = np.asarray(n_hat, dtype=float)

    magmoms = np.zeros((len(atoms), 3))


    A     = primitive.cell[:2, :2]  # 2D lattice vectors from primitive cell
    A_inv = np.linalg.inv(A.T)      # Mapping to lattice coordinates


    for i, atom in enumerate(atoms):
        if atom.symbol not in magnetic_symbols:
            continue

        r_cart = atom.position[:2]  # Cartesian position in 2D plane
        n = A_inv @ r_cart  # Lattice coordinates in primitive basis

        # Phase angle: theta = 2*pi * dot(Q, n) for CLOCKWISE rotation (matching original)
        theta = 2 * np.pi * np.dot(Q[:2], n)
        R = rotation_matrix(n_hat, theta)
        magmoms[i] = R @ m0

    atoms.set_initial_magnetic_moments(magmoms)
    return atoms



def generate_n_hat(theta=0, phi=0):
    """Find cartesian normal vector from spherical angles
    PROVIDE ANGLES IN DEGREES.
    """
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    x = np.cos(phi)*np.sin(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(theta)

    return np.array([x,y,z])


def make_m0_plane(n_hat, magnitude=4.5):
    """Return a vector perpendicular to n_hat with the requested magnitude."""
    n_hat = np.asarray(n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)

    # choose a simple vector not parallel to n_hat
    candidate = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.abs(np.dot(candidate, n_hat)), 1.0, atol=1e-8):
        candidate = np.array([0.0, 1.0, 0.0])

    # project candidate onto the plane perpendicular to n_hat
    m0_plane = candidate - np.dot(candidate, n_hat) * n_hat
    m0_plane = magnitude * m0_plane / np.linalg.norm(m0_plane)
    return m0_plane



def construct_full(theta,phi,path,Q,magnitude,transform=None,magsymbols='Mn', init_moment = None):
    """
    Provide:
    theta: polar angle of normal vector in deg
    phi: azimuthal angle of normal vector in deg
    Q: magnetic ordering vector, e.g. [1/3,1/3,0]
    magnitude: size of magnetic moment
    path: path to cif file
    transform: transformation matrix from primitive to supercell basis.
    magsymbols: symbols of magnetic atoms. Defaults to Mn if nothing provided.
    """
    n = generate_n_hat(theta,phi)
    if init_moment is None:
        m0 = make_m0_plane(n_hat=n, magnitude=magnitude)
    else:
        m0 = np.asarray(init_moment, dtype = float)
        if m0.shape !=(3,):
            raise ValueError ('Init_moment incorrect shape, must be [x,y,z]')

    primitive, supercell, name = cif_reader(path,transform)

    print('n_hat =', n)
    print('Initial moment=' , m0)

    supercell = create_spin_spiral(supercell, primitive, Q, m0, n, magnetic_symbols=magsymbols)

    return supercell, name

