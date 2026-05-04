from gpaw.new.ase_interface import GPAW
from ase.parallel import parprint

name = 'Al2VSe4'

parprint('Now restarting on finer grid')


calc = GPAW('coarse_converged.gpw')
atoms = calc.get_atoms()

calc.set(
    kpts={'size': (12,12,1), 'gamma': True},
    txt=name+'_fine_SCF_GS.txt',
    maxiter=200,
    symmetry='off',
)

atoms.calc = calc
atoms.get_potential_energy()


calc.write(name+'_fine_SCF_GS.gpw')

parprint('Fine grid SCF finished.')
