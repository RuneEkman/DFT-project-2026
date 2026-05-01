from gpaw.new.ase_interface import GPAW
import numpy as np
from ase.dft.kpoints import bandpath
import os
from spinspiral import generate_n_hat

#Bandstructure

# Functions to calculate spin expectation value and orbital decomposition 


def compute_spin_n(calc, e_kn, world, n_vec): #WORKS FOR THE RUNES VERSION OF GPAW
    """
    Compute the spin expectation value <sigma_n> = n_hat dot <sigma>
    for all k-points and bands, where n_hat is an arbitrary unit vector.

    Parameters
    ----------
    calc   : GPAW calculator
    e_kn   : array of eigenvalues (used only for shape)
    world  : MPI communicator
    n_vec  : array-like, shape (3,) — the spin projection direction.
             Does NOT need to be pre-normalised.
             Can be generated from spherical coordinates if you from spinspiral import generate_n_hat

    Returns
    -------
    s_kn : array, shape (nk, nbands), real-valued expectation values.
    """
    # --- normalise direction vector ---
    n_vec = np.asarray(n_vec, dtype=float)
    norm = np.linalg.norm(n_vec)
    if norm < 1e-10:
        raise ValueError("n_vec must be a non-zero vector")
    nx, ny, nz = n_vec / norm

    # --- build n_hat dot sigma as a 2x2 complex matrix ---
    # sigma_x = [[0,1],[1,0]], sigma_y = [[0,-i],[i,0]], sigma_z = [[1,0],[0,-1]]
    # n.sigma = [[ nz,        nx - i*ny ],
    #            [ nx + i*ny, -nz       ]]
    sigma_n = np.array([[ nz,           nx - 1j*ny],
                         [nx + 1j*ny,  -nz         ]], dtype=complex)

    ucvol = np.abs(np.linalg.det(calc.dft.density.nt_sR.desc.cell))
    dO_ii = {a: setup.dO_ii for a, setup in enumerate(calc.dft.setups)}

    s_kn = np.zeros_like(e_kn)

    for wfs in calc.dft.ibzwfs._wfs_u:
        index = wfs.k

        psit_nsG = wfs.psit_nX.data[:]   # shape: (nbands, 2, nG)
        psit1_nG = psit_nsG[:, 0, :]     # spin-up   component
        psit2_nG = psit_nsG[:, 1, :]     # spin-down component

        # Smooth contribution: sum over G of psi† (n.sigma) psi
        # = nz*(|psi1|^2 - |psi2|^2)
        # + (nx-i*ny)*(psi1* psi2)
        # + (nx+i*ny)*(psi2* psi1)
        smooth = (
            sigma_n[0, 0] * np.sum(psit1_nG.conj() * psit1_nG, axis=1)
          + sigma_n[0, 1] * np.sum(psit1_nG.conj() * psit2_nG, axis=1)
          + sigma_n[1, 0] * np.sum(psit2_nG.conj() * psit1_nG, axis=1)
          + sigma_n[1, 1] * np.sum(psit2_nG.conj() * psit2_nG, axis=1)
        ) * ucvol
        s_kn[index] = smooth.real

        # PAW augmentation correction
        for a, P_nsi in wfs.P_ani.items():
            P1_ni = P_nsi[:, 0, :]   # shape: (nbands, nprojs)
            P2_ni = P_nsi[:, 1, :]

            dO = dO_ii[a]

            # Each term: sum_{i,j} P_s1*(i) dO_ij P_s2(j)
            # = einsum('ni,nj,ij->n', P_s1.conj(), P_s2, dO)
            s_kn[index] += (
                sigma_n[0, 0] * np.einsum('ni,nj,ij->n', P1_ni.conj(), P1_ni, dO)
              + sigma_n[0, 1] * np.einsum('ni,nj,ij->n', P1_ni.conj(), P2_ni, dO)
              + sigma_n[1, 0] * np.einsum('ni,nj,ij->n', P2_ni.conj(), P1_ni, dO)
              + sigma_n[1, 1] * np.einsum('ni,nj,ij->n', P2_ni.conj(), P2_ni, dO)
            ).real

    world.sum(s_kn)
    return s_kn


def compute_band_decomposition(calc, e_kn, world): #WORKS FOR THE RUNES VERSION OF GPAW
    """
    Compute per-atom, per-projector PAW weights for all k-points and bands.
    Uses the new GPAW parallel API (calc.dft.ibzwfs) so it works correctly
    across multiple MPI ranks. Results are reduced via world.sum() so every
    rank ends up with the full weight array after the call.
 
    Parameters
    ----------
    calc  : GPAW calculator (new interface)
    e_kn  : (nkpts, nbands) array of eigenvalues already shifted by E_f
    world : MPI communicator
 
    Returns
    -------
    weights     : (nkpts, nbands, natoms, nprojs_max) float array
    projector_l : list of lists — projector_l[a][p] is the l-quantum number
                  of projector p on atom a
    """
    atoms   = calc.get_atoms()
    natoms  = len(atoms)
    symbols = atoms.get_chemical_symbols()
    nkpts, nbands = e_kn.shape
    l_map_inv = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
 
    # --- Projector l-values and nprojs_max from the setups (same on all ranks) ---
    # setup.pt_j gives the projector channels defined in the PAW setup.
    # projector_l[a] is a list of angular momenta for atom a's projectors.
    # This is used later to build orbital-resolved weights.
    setups = calc.dft.setups
    projector_l = []
    nprojs_per_atom = []   # number of projectors per atom from the setup definitions
    for a in range(natoms):
        ls = [phit.l for phit in setups[a].pt_j]
        projector_l.append(ls)
        nprojs_per_atom.append(len(ls))
    nprojs_max = max(nprojs_per_atom)
 
    if world.rank == 0:     #Print statement guarded by world.rank to show which atoms and which projectors
        print("Atom projectors:")
        for a in range(natoms):
            channels = [l_map_inv[l] for l in projector_l[a]]
            print(f"  Atom {a:2d} ({symbols[a]}): {channels} "
                  f"-> {nprojs_per_atom[a]} projectors")
 
    # --- Determine the true maximum number of projectors from the actual P_ani data ---
    # P_ani is the projector coefficient dictionary stored for each k-point wavefunction.
    # The third dimension of P_nsi is the number of projectors actually present.
    # We need the maximum across all ranks so the weights array is large enough everywhere.
    local_nprojs_max = nprojs_max
    for wfs in calc.dft.ibzwfs._wfs_u:
        for P_nsi in wfs.P_ani.values():
            local_nprojs_max = max(local_nprojs_max, P_nsi.shape[2])
    nprojs_max = world.max(local_nprojs_max)
 
    # --- Allocate — initialise to zero so world.sum() is correct ---
    weights = np.zeros((nkpts, nbands, natoms, nprojs_max))
 
    # --- Each rank fills in only the k-points it owns ---
    for wfs in calc.dft.ibzwfs._wfs_u:
        k = wfs.k  # global k-point index owned by this rank
 
        for a, P_nsi in wfs.P_ani.items():
            # P_nsi shape: (nbands, nspinor, nprojs)  — nspinor=2 for SOC
            nprojs = P_nsi.shape[2]
            for proj_idx in range(nprojs):
                # Sum |<psi|p>|^2 over both spinor components
                w = np.sum(np.abs(P_nsi[:, :, proj_idx]) ** 2, axis=1)
                weights[k, :, a, proj_idx] = w.real
 
    # --- MPI reduction: sum contributions from all ranks ---
    world.sum(weights)
 
    return weights

def make_folder():

    n = 0
    folder_name = f"{name}_bandstructure_npoint_{npoints}_run_{n}"

    if os.path.exists(folder_name):
        temp = f"{name}_bandstructure_npoint_{npoints}_run_{n}"
        while os.path.exists(temp):
            n += 1
            temp = f"{name}_bandstructure_npoint_{npoints}_run_{n}"
            

        folder_name = temp
    print('Creating folder:', folder_name)
    os.makedirs(folder_name, exist_ok=True) 
    os.chdir(folder_name) # Change to the new folder for all subsequent file operations


#Parameters that need updating for every material:
#1) Plotting energy window

#Parameters that should be checked before every run:
#1) Npoints
#2) Path
#3) Parallelization choice OBS: this version is fixed to run in parallel!


#Path
M       = [0.5,  0.0, 0.0]
minus_M = [-0.5, 0.0, 0.0]
Gamma   = [0.0,  0.0, 0.0]

kpts = np.array([
    minus_M,
    Gamma,
    M
])

npoints = 1
theta, phi = 0,0

script_dir = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(script_dir) # Ensure we are in the script directory to avoid path issues when running from different locations (Needed for me)

#Load the GS from the SCF run to define the supercell (atoms) and the path
calc_gs = GPAW('MnI2_SCF_GS.gpw', txt=None)
atoms   = calc_gs.get_atoms()
name    = atoms.get_chemical_formula(mode='metal') #saving name as a string for naming text files later.
path    = bandpath(kpts, atoms.cell, npoints=npoints)

# #SOC
# name+= '_SOC_'
# name += 'mode_all'

#-----------Create folder---------------#

make_folder()

#-----------Calc---------------#

print(f'Calculating Band structure, path = {kpts}, npoints={npoints}')
#Non SCF band calc
calc = calc_gs.fixed_density(
    kpts     = path.kpts,
    symmetry = 'off',
    parallel = {'kpt':4,'domain':4,'band':1}, #use 16 cores for parallel in kpt. OBS bands not supported for lcao parallel
    txt      = name+f'_band_npoint{npoints}.txt'
)

calc.get_potential_energy()

#------------saving results---------------------
calc.write(name+f'_band_point{npoints}.gpw', mode='all')    #Enable mode='all' to also save WF's, takes a lot of space.

print('Bands done')
print('Results saved to', name+f'_band_npoint{npoints}')


#communicator for parallel processing
try:
    world = calc.dft.ibzwfs.ibz.comm
except AttributeError:
    from gpaw.mpi import world

#------------Cal and save plotting quantaties---------------------

print('Calculating energy and spin expectation values')

# Get eigenvalues for each k-point and subtract fermi level
ef   = calc.get_fermi_level()
e_kn = np.array([calc.get_eigenvalues(kpt=k) for k in range(len(calc.get_ibz_k_points()))])
e_kn -=ef

bs = calc.band_structure()
bs.energies[...] -= ef # Shift energies by Fermi energy

x      = bs.path.get_linear_kpoint_axis()[0]
X      = bs.path.get_linear_kpoint_axis()[1]
labels = [r'$-\bar{M}$', r'$\Gamma$', r'$\bar{M}$']

#sanity check
assert e_kn.shape[0] == len(x), f"k-point mismatch: {e_kn.shape[0]} vs {len(x)}"


#--------------------spin expectation val----------------------------------

n_hat = generate_n_hat(theta, phi)

s_kn = compute_spin_n(calc, e_kn, world, n_hat)

#--------------------Atomic and orbital projections----------------------------------

weights = compute_band_decomposition(calc, e_kn, world)


#---------saving and plotting only on rank=0-------------------

if world.rank == 0:
    np.save(name + f'_band_npoint{npoints}_energies.npy',e_kn)
    np.save(name + f'_band_npoint{npoints}_spintexture.npy', s_kn)
    np.save(name + f'_band_npoint{npoints}_x.npy', x)
    np.save(name + f'_band_npoint{npoints}_X.npy', X)
    np.save(name + f'_band_npoint{npoints}_orbital_and_atomic_weights.npy', weights) # remember to unforld when plotting 


