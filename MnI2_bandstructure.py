from gpaw.new.ase_interface import GPAW
import numpy as np
from ase.dft.kpoints import bandpath

#Bandstructure

#Parameters that need updating for every material:
#1) Plotting energy window

#Parameters that should be checked before every run:
#1) Npoints
#2) Path
#3) Parallelization choice



#Path
M       = [0.5,  0.0, 0.0]
minus_M = [-0.5, 0.0, 0.0]
Gamma   = [0.0,  0.0, 0.0]

kpts = np.array([
    minus_M,
    Gamma,
    M
])

npoints = 35



#Load the GS from the SCF run to define the supercell (atoms) and the path
calc_gs = GPAW('MnI2_spiral_gs_LONGRUN.gpw', txt=None)
atoms   = calc_gs.get_atoms()
name    = atoms.get_chemical_formula(mode='metal') #saving name as a string for naming text files later.
path    = bandpath(kpts, atoms.cell, npoints=npoints)



#-----------Calc---------------#

print(f'Calculating Band structure, path = {kpts}, npoints={npoints}')
#Non SCF band calc
calc = calc_gs.fixed_density(
    kpts     = path.kpts,
    symmetry = 'off',
    parallel = {'kpt':2,'domain':2,'band':1}, #use 16 cores for parallel in kpt. OBS bands not supported for lcao parallel
    txt      = name+f'_band_npoint{npoints}.txt'
)

calc.get_potential_energy()

#------------saving results---------------------

calc.write(name+f'_band_noṕoint{npoints}.gpw')#, mode='all')    #Enable mode='all' to also save WF's, takes a lot of space.

print('Bands done')
print('Results saved to', name+f'_band_npoint{npoints}')


ef = calc.get_fermi_level()

# Get eigenvalues for each k-point and subtract fermi level
e_kn = np.array([calc.get_eigenvalues(kpt=k) for k in range(len(calc.get_ibz_k_points()))])
e_kn -=ef


print('Now plotting.')

#--------------------plotting----------------------------------

bs = calc.band_structure()
bs.energies[...] -= ef # Shift energies by Fermi energy


import matplotlib.pyplot as plt

bs.plot(filename=name+f'_band_npoint{npoints}_no_color.png',emin=-3.3, emax=0)
# plt.show()
plt.savefig(name+f'_band_npoint{npoints}_no_color.pdf')   



print("Calculating spin expectation value")
#--------------------spin expectation val plot----------------------------------

def compute_spin_z(calc, e_kn):
    """
    Compute the z-component of the spin expectation value <sigma_z>
    for all k-points and bands.
    
    Parameters
    ----------
    calc : GPAW
        A loaded GPAW calculator object with wavefunctions available
    e_kn : np.ndarray
        Array of eigenvalues, shape (nk, nbands)
    
    Returns
    -------
    s_kn : np.ndarray
        Spin expectation values <sigma_z>, shape (nk, nbands),
        values range from -1 (spin down) to +1 (spin up)
    """
    # Unit cell volume
    ucvol = np.abs(np.linalg.det(calc.dft.density.nt_sR.desc.cell))
    
    # PAW overlap correction matrices for each atom
    dO_ii = {a: setup.dO_ii for a, setup in enumerate(calc.dft.setups)}
    
    # Initialize spin array
    s_kn = np.ones_like(e_kn)
    
    for wfs_s in calc.dft.ibzwfs.wfs_qs:
        wfs = wfs_s[0]
        index = wfs.k
        
        # Extract spinor components
        psit_nsG = wfs.psit_nX.data[:]
        psit1_nG = psit_nsG[:, 0, :]  # spin-up
        psit2_nG = psit_nsG[:, 1, :]  # spin-down
        
        # Smooth contribution to <sigma_z>
        s_kn[index] = (np.sum(
            psit1_nG.conj() * psit1_nG - psit2_nG.conj() * psit2_nG,
            axis=1) * ucvol).real
        
        # PAW augmentation correction
        for a, P_nsi in wfs.P_ani.items():
            P1_ni = P_nsi[:, 0, :]
            P2_ni = P_nsi[:, 1, :]
            s_kn[index] += np.einsum('ni,nj,ij->n', P1_ni.conj(), P1_ni, dO_ii[a]).real
            s_kn[index] -= np.einsum('ni,nj,ij->n', P2_ni.conj(), P2_ni, dO_ii[a]).real
    
    # Ensure colormap normalization spans [-1, +1]
    s_kn[0, 0] = 1
    s_kn[0, 1] = -1
    
    return s_kn


s_kn = compute_spin_z(calc, e_kn)

np.savetxt(name+f'_band_npoint{npoints}_spintexture.dat', s_kn)


# Get k-path coordinates and eigenvalues from the band structure object
x      = bs.path.get_linear_kpoint_axis()[0]  # x-axis coordinates
X      = bs.path.get_linear_kpoint_axis()[1]  # high-symmetry point positions
labels = [r'$-\bar{M}$',r'$\Gamma$',r'$\bar{M}$']  # high-symmetry point labels

#Saving the k-points for plotting.
np.savetxt(name+f'_band_npoint{npoints}_x', x)
np.savetxt(name+f'_band_npoint{npoints}_X', X)


#-----------PLOTTING COLORED BANDS-------
from matplotlib.collections import LineCollection

fig, ax = plt.subplots(figsize=(6, 6))

cmap = plt.cm.viridis
vmin, vmax = -1, 1

for n in range(e_kn.shape[1]):  # loop over bands
    # Stack x and y into segments: shape (nk-1, 2, 2)
    points = np.array([x, e_kn[:, n]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Average spin of neighboring k-points as the segment color
    colors = (s_kn[:-1, n] + s_kn[1:, n]) / 2
    
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(vmin, vmax), linewidth=2)
    lc.set_array(colors)
    lc.set_linewidth(1.2)
    ax.add_collection(lc)

# Colorbar
cbar = plt.colorbar(lc, ax=ax, pad=0.02)
cbar.set_label(r'$\langle \sigma_z \rangle$', fontsize=12)

# High-symmetry lines and labels
for xline in X:
    ax.axvline(x=xline, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(X)
ax.set_xticklabels(labels)

# Fermi level
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

ax.set_xlim(x[0], x[-1])
ax.set_ylim(-3, -1)
ax.set_ylabel(r'$E - E_f$ (eV)', fontsize=12)
ax.set_title('Spin-resolved band structure', fontsize=13)

plt.tight_layout()
plt.savefig('spin_bands.png', dpi=150)
plt.savefig('spin_bands.PDF')
# plt.show()