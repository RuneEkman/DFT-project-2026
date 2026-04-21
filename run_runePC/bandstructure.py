from gpaw.new.ase_interface import GPAW
import numpy as np

#Bandstructure

from ase.dft.kpoints import bandpath
M = [0.5, 0.0, 0.0]
minus_M = [-0.5, 0.0, 0.0]
Gamma = [0.0, 0.0, 0.0]

kpts = np.array([
    minus_M,
    Gamma,
    M
])

npoints = 35

#Load the GS from the SCF run to define the supercell (atoms) and the path
calc_gs = GPAW('MnI2_spiral_gs_LONGRUN.gpw', txt=None)
atoms = calc_gs.get_atoms()
path = bandpath(kpts, atoms.cell, npoints=npoints)


print(f'Calculating Band structure, path = {kpts}, npoints={npoints}')
#Non SCF band calc
calc = calc_gs.fixed_density(
    kpts=path.kpts,
    symmetry='off',
    parallel = {'kpt':2,'domain':2,'band':1}, #use 16 cores for parallel in kpt. OBS bands not supported for lcao parallel
    txt = f'band_npoint{npoints}.txt'
)
calc.get_potential_energy()
calc.write('MnI2_bands.gpw', mode='all')

print('Bands done')
print(f'Results should be saved to band_npoint{npoints}.txt')
print('And full info saved to MnI2_bands.gpw')

ef = calc.get_fermi_level()

# Get eigenvalues for each k-point
nk = len(path.kpts)
e_kn = np.array([calc.get_eigenvalues(kpt=k) for k in range(nk)])

#e_kn = np.array([calc.get_eigenvalues(kpt=k) for k in range(len(calc.get_ibz_k_points()))])
e_kn -=ef

# Save to a text file
print(f'Saving eigenvalues minus the fermi level {ef}')
np.savetxt('band_eigvals_MGM_minus_ef.dat', e_kn)

# Get k-points (if you have them in a variable, e.g., 'path')
#np.savetxt('band_kpoints_MGM.dat', path)
np.savetxt('band_kpoints_MGM.dat', path.kpts)

#Save the location of the high-symmetry points
np.savetxt('highsym_MGM.dat' , kpts)

print('Results saved.')
print('Now plotting')

bs = calc.band_structure()
print('Trying to use bs.write, hope it works')
bs.write('bands.dat')
print('it worked')

import matplotlib.pyplot as plt

bs.plot(filename='bands_no_color.png',emin=-3.3, emax=0)
# plt.show()
#plt.savefig('bands_no_color.png')   
plt.savefig('bands_no_color.pdf')   


print('Running TO-script to color bands according to spin')

# from gpaw.spinorbit import soc_eigenstates

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

ef = calc.get_fermi_level()

ucvol = np.abs(np.linalg.det(calc.dft.density.nt_sR.desc.cell))
dO_ii = {}
for a, setup in enumerate(calc.dft.setups):
    dO_ii[a] = setup.dO_ii

#x = np.loadtxt('band_kpoints_MGM.dat')  #NOTE this may be incorrect / might no>
#X = np.loadtxt('highsym_MGM.dat')

#NOTE this is not TO's original script.
# High-symmetry k-points: 0 for -M, npoints//2 for Gamma, npoints-1 for M
X = [0, npoints//2, npoints-1]  # Mark these points as X (
# For the band-path indices:
x = np.arange(len(path.kpts))  # get all k-points' indices


# e_kn -= ef
#for e_k in e_kn.T:                                                            >
#    plt.plot(x, e_k, '--', c='0.5')                                           >
e_kn = np.loadtxt('band_eigvals_MGM_minus_ef.dat')


#soc = soc_eigenstates(calc, scale=0)                                                                                   
#e_kn = soc.eigenvalues()                                                                                               
#e_kn -= ef                                                                                                             
s_kn = np.ones_like(e_kn)

print('Calculating spin texture')

for wfs_s in calc.dft.ibzwfs.wfs_qs:
    wfs = wfs_s[0]
    index = wfs.k
    #kpt_v = wfs.kpt_c @ B_cv                                                                                           
    #e_kn[index] = wfs.eig_n * Ha                                                                                       

    print(index)

    psit_nsG = wfs.psit_nX.data[:]
    psit1_nG = psit_nsG[:, 0, :]
    psit2_nG = psit_nsG[:, 1, :]

    s_kn[index] = (np.sum(psit1_nG.conj() * psit1_nG - psit2_nG.conj() * psit2_nG, axis=1) * ucvol).real
    #s_kn[index] = (np.sum(psit1_nG.conj() * psit2_nG + psit2_nG.conj() * psit1_nG, axis=1) * ucvol).real               

    for a, P_nsi in wfs.P_ani.items():
        P1_ni = P_nsi[:, 0, :]
        P2_ni = P_nsi[:, 1, :]

        s_kn[index] += np.einsum('ni,nj,ij->n', P1_ni.conj(), P1_ni, dO_ii[a]).real
        s_kn[index] -= np.einsum('ni,nj,ij->n', P2_ni.conj(), P2_ni, dO_ii[a]).real
        #s_kn[index] += np.einsum('ni,nj,ij->n', P1_ni.conj(), P2_ni, dO_ii[a]).real                                    
        #s_kn[index] += np.einsum('ni,nj,ij->n', P2_ni.conj(), P1_ni, dO_ii[a]).real                                    

s_kn[0, 0] = 1
s_kn[0, 1] = -1

from gpaw.mpi import world
 #Assembling the spin texture correctly - each core calculates separate parts, so to assemble into one array we do this.
world.barrier()
world.sum(s_kn)
world.barrier()

np.savetxt('spintexture_MGM.dat', s_kn)

print('Spin texture saved')


plt.xticks(X, [r'$\mathrm{-\bar M}$', r'$\Gamma$', r'$\mathrm{\bar M}$'], size=16)
plt.yticks([-2, -1, 0, 1], [-2, -1, 0, 1], size=14)
for i in range(len(X))[1:-1]:
    plt.plot(2 * [X[i]], [1.1 * np.min(e_kn), 1.1 * np.max(e_kn)],
             c='0.5', linewidth=0.5)

things = plt.scatter(np.tile(x, len(e_kn.T)),
                     e_kn.T.reshape(-1),
                     c=s_kn.T.reshape(-1),
                     s=2)
plt.colorbar(things)
plt.ylabel(r'$\varepsilon_n(k)\;\mathrm{[eV]}$', size=24)
plt.axhline(y=0, color='0.5', linestyle='-')
plt.axis([0, x[-1], -2.4, 1.4])
plt.tight_layout()
# plt.show()
plt.savefig('bands_colored.png')
plt.savefig('bands_colored.pdf')   
print('PLot finished, terminating.')
