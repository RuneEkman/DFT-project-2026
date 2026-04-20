rom gpaw.new.ase_interface import GPAW
from ase import Atoms
import numpy as np
from gpaw import FermiDirac
from ase.visualize import view
import ase
from ase.build import make_supercell

primitive = ase.io.read("1MnI2-1.cif")


#Define transformation matrix from primitive to magnetic cell
P = np.array([
    [2, 1, 0],
    [-1, 1, 0],
    [0, 0, 1]
])

supercell = make_supercell(primitive, P)


m = 4.5
magmoms = np.zeros((len(supercell), 3))


A = primitive.cell[:2, :2] #extracts the 2D lattice vectors from the PRIMITIVE CELL. It is a 2x2 matrix.
A_inv = np.linalg.inv(A.T) #Invert the above, constructs a mapping from cartesian to lattice coordinates. Transposed since AS>
                            #This gives \vec{n}=A^{-1}\vec{r}, where r is cartesian position, n is coordinates in lattice bas>
Q = np.array([1/3, 1/3])    #Defines the magnetic ordering vector in reciprocal lattice coordinates.

for i, atom in enumerate(supercell):
    if atom.symbol != 'Mn':
        continue

    r_cart = atom.position[:2] #extracts the cartesian position of the Mn atom
    n = A_inv @ r_cart         # Converts to lattice coordinates \vec{r}=n_1a_1+n_2a_2

    phase = 2 * np.pi * np.dot(-Q, n) #each lattice site gets a phase angle depending on its position (given via the lattice >
    #NOTE minus Q above for clockwise

    magmoms[i] = [                   #Assigns a planar spin spiral, note that m_0=(0,m,0) to get this.
        -m * np.sin(phase),
        m * np.cos(phase),
        0.0
    ]


supercell.set_initial_magnetic_moments(magmoms)



# --- 4. Setup GPAW Calculator ---
# Article specifies: LDA functional, 600 eV cutoff.
# Symmetry must be off for non-collinear spirals.
# k-points: The magnetic BZ is smaller.
# Let's use a dense grid to resolve the bands smoothly for Fig 1.
calc = GPAW(
    mode={'name':'pw',
          'ecut':600},          # 600 eV cutoff in per paper. Rough first guess
    xc='LDA',              # Paper explicitly uses LDA 
    mixer={'backend': 'pulay',              #This was used to mimic https://gpaw.readthedocs.io/tutorialsexercises/magnetic/s>
                       'beta': 0.05,
                       'method': 'sum',
                       'nmaxold': 5,
                       'weight': 100},
    kpts={'size':(12,12,1), 'gamma':True},	 # Adjusted for the rhombus supercell
    symmetry='off',	   # Crucial for spiral
    magmoms=magmoms,	   # Enforce non-collinear start
    spinpol=True,          # Needed for non-collinear
    occupations=FermiDirac(0.01),
    txt='MnI2_spiral_supercell_attempt2.txt',
    maxiter=100,
    parallel={'domain':4,'kpt':4,'band':1} # Attempt at running in parallel for the compute node.
)

calc.verbosity=1

supercell.calc = calc


# --- 5. Run SCF Calculation ---
print("Running SCF for magnetic supercell...")
energy = supercell.get_potential_energy()
calc.write('MnI2_spiral_gs_attempt2.gpw',mode='all')
print('Finished SCF calculation, result saved in gpw file.')

#Checking convergence and magnetic moments
energy = supercell.get_potential_energy()
print('energy=',energy) #check for divergence. If it diverges, something wrong in the setup
print('')
#Magmoms
print('Local magnetic moments = ', calc.get_non_collinear_magnetic_moments())
#We want three Mn moments, roughly 120 degrees apart, magnitude around 3-5 µB
print('')
#Total M
print('Total magnetic moment=', supercell.get_magnetic_moment())
#Should be 0. If they align ferromagnetically, phase may be incorrectly assigned


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

npoints = 100

path = bandpath(kpts, supercell.cell, npoints=npoints)

print(f'Calculating Band structure, path = {kpts}, npoints={npoints}')
#Non SCF band calc
calc = GPAW('MnI2_spiral_gs.gpw').fixed_density(
    kpts=path,
    symmetry='off',
    parallel = {'kpt':1,'domain':4,'band':1}, #use 4 cores for parallel in kpt. OBS bands not supported for lcao parallel
)

print('Bands done')
print('')

ef = calc.get_fermi_level()

# Get eigenvalues for each k-point
e_kn = np.array([calc.get_eigenvalues(kpt=k) for k in range(len(calc.get_ibz_k_points()))])
e_kn -=ef

# Save to a text file
np.savetxt('band_eigvals_MGM_minus_ef.dat', e_kn)

# Get k-points (if you have them in a variable, e.g., 'path')
np.savetxt('band_kpoints_MGM.dat', path)

#Save the location of the high-symmetry points
np.savetxt('highsym_MGM.dat' , kpts)

print('Results saved.')
print('Now plotting')

bs = calc.band_structure()
print('Trying to use bs.write, hope it works')
bs.write('bands.dat')
print('it worked')

import matplotlib.pyplot as plt

bs.plot(fermi = calc.get_fermi_level(),emin=-3.3, emax=0)
# plt.show()
plt.savefig('bands_no_color.png')   
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

x = np.loadtxt('band_kpoints_MGM.dat')
X = np.loadtxt('highsym_MGM.dat')
# e_kn = np.array([calc.get_eigenvalues(kpt=k)
                #  for k in range(len(calc.get_ibz_k_points()))])

# e_kn -= ef
#for e_k in e_kn.T:
#    plt.plot(x, e_k, '--', c='0.5')
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

