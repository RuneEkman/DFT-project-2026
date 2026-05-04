from gpaw.new.ase_interface import GPAW
from ase import Atoms
import numpy as np
from gpaw import FermiDirac
from ase.parallel import parprint


from pathlib import Path
import sys

# Add the parent directory to the path
sys.path.append(str(Path().resolve().parent))

# Now import
from spinspiral import construct_full


#NOTE this is WITH SOC!
#NOTE I have RETIGHTENED the mixer.


#Normal vector orientation in spherical
theta, phi = 90,0

path_to_cif = "1VAl2Se4-1.cif"

Q = [1/3,1/3,0]

magnetic_magnitude = 2.4

#Define transformation matrix from primitive to magnetic cell
P = np.array([
    [2, 1, 0],
    [-1, 1, 0],
    [0, 0, 1]
])

magnetic_atom = 'V'

supercell , name = construct_full(theta = theta, phi=phi, Q=Q , path= path_to_cif, transform=P, magnitude=magnetic_magnitude, magsymbols=magnetic_atom)
#
#, init_moment=[0,4.5,0])
#Note: init_moment can be specified to mimic the article by TO.

name += '_SOC_'

magmoms = supercell.arrays['initial_magmoms']

# ---  Setup GPAW Calculator ---
# Article specifies: LDA functional, 600 eV cutoff.
# Symmetry must be off for non-collinear spirals.
# k-points: The magnetic BZ is smaller.
calc = GPAW(
    mode={'name':'pw',
          'ecut':600},          # 600 eV cutoff in per paper. 
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
    txt=name+'_SCF_GS.txt',
    maxiter=200,
    parallel={'domain':4,'kpt':4,'band':1}, # Attempt at running in parallel for the compute node.
    soc= True,
    #convergence={'density': 1e-9, 'energy': 5e-7, 'eigenstates': 1e-10}  # Tightened criteria
)

#Old behaviour:
#    mixer={'backend': 'pulay',              #This was used to mimic https://gpaw.readthedocs.io/tutorialsexercises/magnetic/s>
#                       'beta': 0.05,
#                       'method': 'sum',
#                       'nmaxold': 5,
#                       'weight': 100},
#kpts = {'size':(12,12,1), 'gamma':True},	 # Adjusted for the rhombus supercell


calc.verbosity=1

supercell.calc = calc


# --- 5. Run SCF Calculation ---
parprint("Running SCF for magnetic supercell...")
energy = supercell.get_potential_energy()
calc.write('coarse_converged.gpw')#,mode='all')
parprint('Finished SCF calculation, result saved in gpw file.')

#Checking convergence and magnetic moments
energy = supercell.get_potential_energy()
parprint('energy=',energy) #check for divergence. If it diverges, something wrong in the setup
parprint('')
#Magmoms
parprint('Local magnetic moments = ', calc.get_non_collinear_magnetic_moments())
#We want three Mn moments, roughly 120 degrees apart, magnitude around 3-5 µB
parprint('')
#Total M
parprint('Total magnetic moment=', supercell.get_magnetic_moment())
#Should be 0. If they align ferromagnetically, phase may be incorrectly assigned


# parprint('Now restarting on finer grid')


# calc = GPAW('coarse_converged.gpw')
# atoms = calc.get_atoms()

# calc.set(
#     kpts={'size': (12,12,1), 'gamma': True},
#     mixer={'backend': 'pulay',              #This was used to mimic https://gpaw.readthedocs.io/tutorialsexercises/magnetic/s>
#                        'beta': 0.10,
#                        'method': 'sum',
#                        'nmaxold': 5,
#                        'weight': 50},
#     txt=name+'_fine_SCF_GS.txt'
# )

# atoms.calc = calc
# atoms.get_potential_energy()


# calc.write(name+'_fine_SCF_GS.gpw')

# parprint('Fine grid SCF finished.')
