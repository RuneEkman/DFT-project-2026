from gpaw.new.ase_interface import GPAW
import numpy as np
from ase.dft.kpoints import bandpath
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

#Bandstructure

#Parameters that need updating for every material:
#1) Plotting energy window
#2) GS SCF file import

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
        index = wfs.k

        psit_nsG = wfs.psit_nX.data[:]
        psit1_nG = psit_nsG[:, 0, :]  # spin-up
        psit2_nG = psit_nsG[:, 1, :]  # spin-down

        # Smooth contribution
        s_kn[index] = (np.sum(
            psit1_nG.conj() * psit1_nG - psit2_nG.conj() * psit2_nG,
            axis=1) * ucvol).real

        # PAW augmentation correction
        for a, P_nsi in wfs.P_ani.items():
            P1_ni = P_nsi[:, 0, :]
            P2_ni = P_nsi[:, 1, :]
            s_kn[index] += np.einsum('ni,nj,ij->n', P1_ni.conj(), P1_ni, dO_ii[a]).real
            s_kn[index] -= np.einsum('ni,nj,ij->n', P2_ni.conj(), P2_ni, dO_ii[a]).real

    # MPI reduction: sum across all ranks
    world.sum(s_kn)

    # Colormap anchors — set after reduction
    s_kn[0, 0] =  1
    s_kn[0, 1] = -1

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
    vmin, vmax = -1, 1

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
    plt.savefig('spin_bands.png', dpi=150)
    plt.savefig('spin_bands.pdf')

    print("Done. Results saved to", name + f'_band_npoint{npoints}')