import numpy as np                                                                                                      
from gpaw.occupations import create_occ_calc                                                                            
from gpaw.spinorbit import soc_eigenstates                                                                              
from gpaw.new.ase_interface import GPAW   


def sphere_points(distance=None):                                                                                       
    '''Calculates equidistant points on the upper half sphere                                                           
                                                                                                                        
    Returns list of spherical coordinates (thetas, phis) in degrees                                                     
                                                                                                                        
    Modified from:                                                                                                      
        M. Deserno 2004 If Polymerforshung (Ed.) 2 99                                                                   
    '''                                                                                                                 
                                                                                                                        
    import math                                                                                                         
    N = math.ceil(129600 / (math.pi) * 1 / distance**2)                                                                 
    if N <= 1:                                                                                                          
        return np.array([0.]), np.array([0.])                                                                           
                                                                                                                        
    A = 4 * math.pi                                                                                                     
    a = A / N                                                                                                           
    d = math.sqrt(a)                                                                                                    
                                                                                                                        
    # Even number of theta angles ensure 90 deg is included                                                             
    Mtheta = round(math.pi / (2 * d)) * 2                                                                               
    dtheta = math.pi / Mtheta                                                                                           
    dphi = a / dtheta                                                                                                   
    points = []                                                                                                         
                                                                                                                        
    # Limit theta loop to upper half-sphere                                                                             
    for m in range(Mtheta // 2 + 1):                                                                                    
        # m = 0 ensure 0 deg is included, Mphi = 1 is used in this case                                                 
        theta = math.pi * m / Mtheta                                                                                    
        Mphi = max(round(2 * math.pi * math.sin(theta) / dphi), 1)                                                      
        for n in range(Mphi):                                                                                           
            phi = 2 * math.pi * n / Mphi                                                                                
            points.append([theta, phi])                                                                                 
    thetas, phis = np.array(points).T                                                                                   
                                                                                                                        
    if not any(thetas - np.pi / 2 < 1e-14):                                                                             
        import warnings                                                                                                 
        warnings.warn('xy-plane not included in sampling')                                                              
                                                                                                                        
    return thetas * 180 / math.pi, phis * 180 / math.pi        


def stereo_project_point(inpoint, axis=0, r=1):                                                                         
    point = np.divide(inpoint * r, inpoint[axis] + r)                                                                   
    point[axis] = 0                                                                                                     
    return point          



def plot_circle(socdata, name='SOC_plot',save=True,):
    from matplotlib.colors import Normalize                                                                                 
    from matplotlib import pyplot as plt                                                                                    
    from scipy.interpolate import griddata                                                                                  
    import numpy as np                                                                                                      
    
    # Load data from nii2_soc.py                                                                                            
    data = np.load(socdata)                                                                                          
    theta, phi = data['theta'], data['phi']                                                                                 
    soc = (data['soc'] - min(data['soc'])) * 10**3                                                                          
                                                                                                                            
    # Convert angles to xyz coordinates                                                                                     
    theta = theta * np.pi / 180                                                                                             
    phi = phi * np.pi / 180                                                                                                 
    x = np.sin(theta) * np.cos(phi)                                                                                         
    y = np.sin(theta) * np.sin(phi)                                                                                         
    z = np.cos(theta)                                                                                                       
    points = np.array([x, y, z]).T           

    # Detect hemisphere and set projection pole accordingly:
    #   upper (z >= 0): project from south pole (r=+1, default)
    #   lower (z <= 0): project from north pole (r=-1)
    lower_hemisphere = np.mean(z) < 0
    r = -1 if lower_hemisphere else 1

    projected_points = [stereo_project_point(p, axis=2, r=r) for p in points]

    #PLOT                                                
                                                                                                                            
    fig, ax = plt.subplots(1, 1, figsize=(5 * 1.25, 5))                                                                     
                                                                                                                            
    # Plot contour surface                                                                                                  
    norm = Normalize(vmin=min(soc), vmax=max(soc))                                                                          
    X, Y, Z = np.array(projected_points).T                                                                                  
    xi = np.linspace(min(X), max(X), 100)                                                                                   
    yi = np.linspace(min(Y), max(Y), 100)                                                                                   
    zi = griddata((X, Y), soc, (xi[None, :], yi[:, None]))                                                                  
    ax.contour(xi, yi, zi, 15, linewidths=0.5, colors='k', norm=norm)                                                       
    ax.contourf(xi, yi, zi, 15, cmap=plt.cm.jet, norm=norm)                                                                 
                                                                                                                            
    # Add additional contours                                                                                               
    mask = np.argwhere(soc <= np.min(soc) + 0.05)                                                                           
    #ax.scatter(X[mask], Y[mask], marker='o', c='midnightblue', s=5)                                                        
    mask = np.argwhere(soc <= np.min(soc) + 0.001)                                                                          
    #ax.scatter(X[mask], Y[mask], marker='o', c='k', s=5)                                                                   
    # Spin-orbit energy minimum                                                                                             
    mask = np.argwhere(soc <= np.min(soc))                                                                                  
    #ax.scatter(X[mask], Y[mask], marker='o', c='white', s=5)                                                               
    # z-axis direction                                                                                                      
    #ax.scatter(X[0], Y[0], marker='o', c='k', s=10)                                                                        
                                                                                                                            
    theta_min = round(theta[mask][0][0] * 180 / np.pi, 2)                                                                   
    phi_min = round(phi[mask][0][0] * 180 / np.pi, 2)                                                                       
    print(f'n = (theta, phi) = ({theta_min}, {phi_min})')                                                                   
                                                                                                                            
    # Set plot details                                                                                                      
    ax.axis('equal')                                                                                                        
    ax.set_xlim(-1.05, 1.05)                                                                                                
    ax.set_ylim(-1.05, 1.05)                                                                                                
    ax.set_xticks([])                                                                                                       
    ax.set_yticks([])                                                                                                       
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax)                                                
    cbar.ax.set_ylabel(r'$E_{soc} [meV]$')                                                                                  
                                                                                                                            
    # Save figure   
    if save:
        plt.savefig(name)                                                                                                         
    plt.show()
                                                                                                   


def sphere_points_lower(distance=None):
    '''Calculates equidistant points on the lower half sphere,
    corresponding to opposite chirality spin spiral configurations.

    Returns list of spherical coordinates (thetas, phis) in degrees.
    Mirrors sphere_points() by reflecting theta -> 180 - theta.
    '''
    thetas, phis = sphere_points(distance=distance)
    return 180.0 - thetas, phis
                                                                                                