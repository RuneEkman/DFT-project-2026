from gpaw.new.ase_interface import GPAW   
from gpaw.occupations import create_occ_calc                                                                            
from gpaw.spinorbit import soc_eigenstates  
import numpy as np

from circleplots import sphere_points_lower

theta_tp, phi_tp = sphere_points_lower(distance=5)                                                                            
calc = GPAW('../MnI2_SCF_GS.gpw')                                                                                             
occcalc = create_occ_calc({'name': 'fermi-dirac', 'width': 0.001})                                                      
soc_tp = np.array([])                                                                                                   
for theta, phi in zip(theta_tp, phi_tp):                                                                                
    en_soc = soc_eigenstates(calc=calc, projected=False, theta=theta, phi=phi,                                          
                             occcalc=occcalc).calculate_band_energy()                                                   
    soc_tp = np.append(soc_tp, en_soc)                                                                                  
                                                                                                                        
np.savez('soc_data_opposite_chiral.npz', soc=soc_tp, theta=theta_tp, phi=phi_tp)                                           