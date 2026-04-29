from gpaw.new.ase_interface import GPAW
import numpy as np
from ase.dft.kpoints import bandpath
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from matplotlib import colormaps
import matplotlib.colors as mcolors

#Bandstructure

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

npoints = 100



#Load the GS from the SCF run to define the supercell (atoms) and the path
calc_gs = GPAW('MnI2_SOC__SCF_GS.gpw', txt=None)
atoms   = calc_gs.get_atoms()
name    = atoms.get_chemical_formula(mode='metal') #saving name as a string for naming text files later.
path    = bandpath(kpts, atoms.cell, npoints=npoints)


#SOC
name+= '_SOC_'

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

calc.write(name+f'_band_point{npoints}.gpw')#, mode='all')    #Enable mode='all' to also save WF's, takes a lot of space.

print('Bands done')
print('Results saved to', name+f'_band_npoint{npoints}')


#communicator for parallel processing
try:
    world = calc.dft.ibzwfs.ibz.comm
except AttributeError:
    from gpaw.mpi import world

# Get eigenvalues for each k-point and subtract fermi level
ef   = calc.get_fermi_level()
e_kn = np.array([calc.get_eigenvalues(kpt=k) for k in range(len(calc.get_ibz_k_points()))])
e_kn -=ef

bs = calc.band_structure()
bs.energies[...] -= ef # Shift energies by Fermi energy

#sanity check
x      = bs.path.get_linear_kpoint_axis()[0]
X      = bs.path.get_linear_kpoint_axis()[1]
labels = [r'$-\bar{M}$', r'$\Gamma$', r'$\bar{M}$']

assert e_kn.shape[0] == len(x), f"k-point mismatch: {e_kn.shape[0]} vs {len(x)}"


print('Now plotting.')

#--------------------plotting----------------------------------
if world.rank == 0:
    bs.plot(filename=name + f'_band_npoint{npoints}_no_color.png', emin=-3.3, emax=0)
    plt.savefig(name + f'_band_npoint{npoints}_no_color.pdf')


print("Calculating spin expectation value")

#--------------------spin expectation val plot----------------------------------
def compute_spin_z(calc, e_kn, world):
    """
    Compute the z-component of the spin expectation value <sigma_z>
    for all k-points and bands. Works with MPI parallelism (PWFD mode).
    """
    ucvol = np.abs(np.linalg.det(calc.dft.density.nt_sR.desc.cell))
    dO_ii = {a: setup.dO_ii for a, setup in enumerate(calc.dft.setups)}

    # Initialize to zero so the allreduce sum is correct
    s_kn = np.zeros_like(e_kn)

    for wfs in calc.dft.ibzwfs._wfs_u:   # <-- flat list, no [0] needed
        # wfs is the wavefunction object for one k-point on this MPI rank.
        # wfs.k is the global k-point index in the irreducible BZ.
        index = wfs.k

        # The spinor wavefunction data is stored in psit_nX with spin index first.
        psit_nsG = wfs.psit_nX.data[:]
        psit1_nG = psit_nsG[:, 0, :]  # spin-up component
        psit2_nG = psit_nsG[:, 1, :]  # spin-down component

        # Smooth (plane-wave) contribution to <sigma_z>.
        s_kn[index] = (np.sum(
            psit1_nG.conj() * psit1_nG - psit2_nG.conj() * psit2_nG,
            axis=1) * ucvol).real

        # PAW augmentation correction from projector overlaps.
        # P_ani stores atomic projector coefficients for each band and spinor.
        # Its shape is (nbands, nspinor, nprojs).
        for a, P_nsi in wfs.P_ani.items():
            P1_ni = P_nsi[:, 0, :]
            P2_ni = P_nsi[:, 1, :]
            s_kn[index] += np.einsum('ni,nj,ij->n', P1_ni.conj(), P1_ni, dO_ii[a]).real
            s_kn[index] -= np.einsum('ni,nj,ij->n', P2_ni.conj(), P2_ni, dO_ii[a]).real

    # MPI reduction: sum across all ranks
    world.sum(s_kn)

    return s_kn


s_kn = compute_spin_z(calc, e_kn, world)

#s_kn = compute_spin_z(calc, e_kn)



#---------saving and plotting only on rank=0-------------------
if world.rank == 0:
    np.savetxt(name + f'_band_npoint{npoints}_spintexture.dat', s_kn)
    np.savetxt(name + f'_band_npoint{npoints}_x', x)
    np.savetxt(name + f'_band_npoint{npoints}_X', X)

    fig, ax = plt.subplots(figsize=(6, 6))

    cmap = plt.cm.viridis
    # vmin, vmax = -1, 1
    vmin, vmax = np.min(s_kn) , np.max(s_kn)

    for n in range(e_kn.shape[1]):
        points   = np.array([x, e_kn[:, n]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors   = (s_kn[:-1, n] + s_kn[1:, n]) / 2

        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(vmin, vmax))
        lc.set_array(colors)
        lc.set_linewidth(1.2)
        ax.add_collection(lc)

    cbar = plt.colorbar(lc, ax=ax, pad=0.02)
    cbar.set_label(r'$\langle \sigma_z \rangle$', fontsize=12)

    for xline in X:
        ax.axvline(x=xline, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(X)
    ax.set_xticklabels(labels)

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-3, -1)
    ax.set_ylabel(r'$E - E_f$ (eV)', fontsize=12)
    ax.set_title('Spin-resolved band structure', fontsize=13)

    plt.tight_layout()
    plt.savefig(name+'spin_bands.png', dpi=150)
    plt.savefig(name+'spin_bands.pdf')

    print("Done. Results saved to", name + f'_band_npoint{npoints}')


#------------Adding orbital projection code--------------------
#Currently set up for parallel processing

################## Plot the orbital/atomic decomposition of the bands using the projector weights ##############
def compute_band_decomposition(calc, e_kn, world):
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
    meta        : dict with atom/band bookkeeping info
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
 
    meta = {
        'e_fermi'   : calc.get_fermi_level(),
        'symbols'   : symbols,
        'natoms'    : natoms,
        'nbands'    : nbands,
        'nkpts'     : nkpts,
        'l_map_inv' : l_map_inv,
        'mn_indices': [i for i, s in enumerate(symbols) if s == 'Mn'],
        'i_indices' : [i for i, s in enumerate(symbols) if s == 'I'],
    }
 
    return weights, projector_l, meta
 
 
def weight_by_atom_group(weights, atom_indices):
    """Sum weights over a group of atoms and all their projectors."""
    return weights[:, :, atom_indices, :].sum(axis=(2, 3))  # shape (nkpts, nbands)
 
def weight_by_l(weights, projector_l, atom_indices, l_target):
    """Sum weights over a group of atoms for a specific l channel."""
    w = np.zeros(weights.shape[:2])  # shape (nkpts, nbands)
    for a in atom_indices:
        for proj_idx, l in enumerate(projector_l[a]):
            if l == l_target:
                w += weights[:, :, a, proj_idx]
    return w  # shape (nkpts, nbands)
 
 
print("Calculating orbital decomposition")
 
# --- Run on all ranks; world.sum() inside ensures all ranks get the full result ---
weights, projector_l, meta = compute_band_decomposition(calc, e_kn, world)
 
mn_idx = meta['mn_indices']
i_idx  = meta['i_indices']
 
# Atomic decomposition — shape (nkpts, nbands)
w_mn = weight_by_atom_group(weights, mn_idx)
w_i  = weight_by_atom_group(weights, i_idx)
 
# Orbital decomposition on Mn — shape (nkpts, nbands)
w_mn_s = weight_by_l(weights, projector_l, mn_idx, l_target=0)
w_mn_p = weight_by_l(weights, projector_l, mn_idx, l_target=1)
w_mn_d = weight_by_l(weights, projector_l, mn_idx, l_target=2)
 
# Orbital decomposition on I — shape (nkpts, nbands)
w_i_s  = weight_by_l(weights, projector_l, i_idx, l_target=0)
w_i_p  = weight_by_l(weights, projector_l, i_idx, l_target=1)
w_i_d  = weight_by_l(weights, projector_l, i_idx, l_target=2)
 
 
#---------saving and plotting only on rank=0-------------------
if world.rank == 0:
    # Save raw weights (4-D array reshaped to 2-D for savetxt)
    nkpts_s, nbands_s, natoms_s, nprojs_max_s = weights.shape
    np.savetxt(name + f'_band_npoint{npoints}_orbital_and_atomic_weights.dat',
               weights.reshape(nkpts_s * nbands_s, natoms_s * nprojs_max_s))
 
    def plot_colored_segments(ax, x_axis, y, w, cmap_name, vmin=0, vmax=1):
        cmap = colormaps[cmap_name]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        for i in range(len(x_axis) - 1):
            w_mid = 0.5 * (w[i] + w[i + 1])
            ax.plot([x_axis[i], x_axis[i + 1]], [y[i], y[i + 1]],
                    color=cmap(norm(w_mid)), lw=2)
        return cm.ScalarMappable(norm=norm, cmap=cmap)
 
    def plot_projection(axes_list, panel_labels, eigs, weight_list,
                        band_indices, cmap_list, ylim=(-2.5, -0.5)):
        """
        axes_list   : list of matplotlib axes, one per projection
        panel_labels: list of strings, one per projection
        weight_list : list of (nkpts, nbands) arrays, one per projection
        band_indices: list of band indices to plot
 
        Uses the existing x / X / labels arrays from the outer scope so that
        the k-axis matches the spin-texture plot exactly.
        """
        for ax, label, w_all, cmap_name in zip(axes_list, panel_labels,
                                                weight_list, cmap_list):
            vmax = w_all[:, band_indices].max()
            sm = None
            for b in band_indices:
                sm = plot_colored_segments(ax, x, eigs[:, b], w_all[:, b],
                                           cmap_name, vmin=0, vmax=vmax)
            plt.colorbar(sm, ax=ax, label='weight')
            ax.set_title(label)
            for xline in X:
                ax.axvline(x=xline, color='black', linewidth=0.8, linestyle='--')
            ax.set_xticks(X)
            ax.set_xticklabels(labels)
            ax.set_xlim(x[0], x[-1])
            ax.set_xlabel(r'$k$')
            ax.set_ylim(ylim)
 
    band_indices = list(range(67, 87))  # adjust to bands of interest
 
    # --- Atomic decomposition plot ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    axes[0].set_ylabel(r'$E - E_F$ (eV)')
    plot_projection(axes, ['Mn', 'I'], e_kn,
                    [w_mn, w_i], band_indices, ['Purples', 'Oranges'])
    fig.suptitle('Atomic decomposition')
    plt.tight_layout()
    plt.savefig('Atomic_decomposition.png', dpi=150)
    plt.savefig('Atomic_decomposition.pdf', bbox_inches='tight')
 
    # --- Orbital decomposition on Mn ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    axes[0].set_ylabel(r'$E - E_F$ (eV)')
    plot_projection(axes, ['Mn-s', 'Mn-p', 'Mn-d'], e_kn,
                    [w_mn_s, w_mn_p, w_mn_d], band_indices,
                    ['Blues', 'Greens', 'Reds'])
    fig.suptitle('Orbital decomposition — Mn')
    plt.tight_layout()
    plt.savefig('Orbital_decomposition_Mn.png', dpi=150)
    plt.savefig('Orbital_decomposition_Mn.pdf', bbox_inches='tight')
 
    # --- Orbital decomposition on I ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    axes[0].set_ylabel(r'$E - E_F$ (eV)')
    plot_projection(axes, ['I-s', 'I-p'], e_kn,
                    [w_i_s, w_i_p], band_indices,
                    ['Blues', 'Greens'])
    fig.suptitle('Orbital decomposition — I')
    plt.tight_layout()
    plt.savefig('Orbital_decomposition_I.png', dpi=150)
    plt.savefig('Orbital_decomposition_I.pdf', bbox_inches='tight')
 
    print("Orbital decomposition done.")
 
