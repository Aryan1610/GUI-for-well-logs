import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

def VRH(volumes,M):
    """Computes Voigt, Reuss, and Hill Average Moduli Estimate.

    Parameters
    ----------
    volumes : list or array-like
        volumetric fractions of N phases
    M : list or array-like
        elastic modulus of the N phase.

    Returns
    -------
    float
        M_v: Voigt average
        M_r: Reuss average
        M_0: Hill average
    """        

    volumes = np.asanyarray(volumes)
    
    M=np.asanyarray(M)
    M_v=np.dot(volumes,M)
    M_r=np.dot(volumes,1/M)**-1
    M_h= 0.5*(M_r+M_v)
    return  M_v,M_r,M_h

def cripor(K0, G0, phi, phic):
    """Critical porosity model according to Nur’s modified Voigt average.

    Parameters
    ----------
    K0 : float or array-like
        mineral bulk modulus in GPa
    G0 : float or array-like
        mineral shear modulus in GPa
    phi : float or array-like
        porosity in frac
    phic : float
        critical porosity in frac

    Returns
    -------
    float or array-like
        K_dry,G_dry (GPa): dry elastic moduli of the framework
    """        

    K_dry = K0 * (1-phi/phic)
    G_dry = G0 * (1-phi/phic)

    return K_dry , G_dry

def cripor_reuss(M0, Mf, phic, den=False):
    """In the suspension domain, the effective bulk and shear moduli of the rock can be estimated by using the Reuss (isostress) average.

    Parameters
    ----------
    M0 : float 
        The solid phase modulus or density
    Mf : float
        The pore filled phase modulus or density
    phic : float
        critical porosity
    den : bool, optional
        If False: compute the reuss average for effective modulus of two mixing phases. If true, compute avearge density using mass balance, which corresponds to voigt average. Defaults to False.

    Returns
    -------
    float or array-like
        M (GPa/g.cc): average modulus or average density

    References
    ----------
    - Section 7.1 Rock physics handbook 2nd edition
    """        

    if den is False:

        M = VRH( np.array([(1-phic,phic)]), np.array([M0,Mf]))[1]
    else:
        M = VRH(np.array([(1-phic,phic)]),np.array([M0,Mf]))[0]

    return M

def hashin_shtrikman(f, k, mu, modulus='bulk'):
    """
    Hashin-Shtrikman bounds for a mixture of two constituents.
    The best bounds for an isotropic elastic mixture, which give
    the narrowest possible range of elastic modulus without
    specifying anything about the geometries of the constituents.

    Args:
        f: list or array of volume fractions (must sum to 1.00 or 100%).
        k: bulk modulus of constituents (list or array).
        mu: shear modulus of constituents (list or array).
        modulus: A string specifying whether to return either the
            'bulk' or 'shear' HS bound.

    Returns:
        namedtuple: The Hashin Shtrikman (lower, upper) bounds.

    :source: Berryman, J.G., 1993, Mixture theories for rock properties
             Mavko, G., 1993, Rock Physics Formulas.

    : Written originally by Xingzhou 'Frank' Liu, in MATLAB
    : modified by Isao Takahashi, 4/27/99,
    : Translated into Python by Evan Bianco
    """
    def z_bulk(k, mu):
        return (4/3.) * mu

    def z_shear(k, mu):
        return mu * (9 * k + 8 * mu) / (k + 2 * mu) / 6

    def bound(f, k, z):
        return 1 / sum(f / (k + z)) - z

    f = np.array(f)
    if sum(f) == 100:
        f /= 100.0

    func = {'shear': z_shear,
            'bulk': z_bulk}

    k, mu = np.array(k), np.array(mu)
    z_min = func[modulus](np.amin(k), np.amin(mu))
    z_max = func[modulus](np.amax(k), np.amax(mu))

    fields = ['lower_bound', 'upper_bound']
    HashinShtrikman = namedtuple('HashinShtrikman', fields)
    return HashinShtrikman(bound(f, k, z_min), bound(f, k, z_max))

def HS(f, K1, K2,G1, G2, bound='upper'):
    """Compute effective moduli of two-phase composite using hashin-strikmann bounds.

    Parameters
    ----------
    f : float
        0-1, volume fraction of stiff material
    K1 : float or array-like
        bulk modulus of stiff phase
    K2 : float or array-like
        bulk modulus of soft phase
    G1 : float or array-like
        shear modulus of stiff phase
    G2 : float or array-like
        shear modulus of soft phase
    bound : str, optional
        upper bound or lower bound. Defaults to 'upper'.

    Returns
    -------
    float or array-like
        K, G (GPa): effective moduli of two-phase composite
    """        

    if bound == 'upper':
        K=K1+ (1-f)/( (K2-K1)**-1 + f*(K1+4*G1/3)**-1 )

        Temp = (K1+2*G1)/(5*G1 *(K1+4*G1/3))
        G=G1+(1-f)/( (G2-G1)**-1 + 2*f*Temp)
    else:
        K=K2+ f/( (K1-K2)**-1 + (1-f)*(K2+4*G2/3)**-1 )
#         not working
#         Temp = (K2+2*G2)/(5*G2 *(K2+4*G2/3))
#         G=G2+f/( (G1-G2)**-1 + 2*(1-f)*Temp)
        G=0
    return K, G

def Gassmann(K_dry,G_dry,K_mat,Kf,phi):
    """Computes saturated elastic moduli of rock via Gassmann equation given dry-rock moduli. 

    Parameters
    ----------
    K_dry : float or array-like
        dry frame bulk modulus 
    G_dry : float or array-like
        dry frame shear modulus 
    K_mat : float
        matrix bulk modulus
    Kf : float
        fluid bulk modulus
    phi : float or array-like
        porosity

    Returns
    -------
    float or array-like
        K_sat, G_sat: fluid saturated elastic moduli
    """        
        
    A=(1-K_dry/K_mat)**2
    B=phi/Kf+(1-phi)/K_mat-K_dry/(K_mat**2)
    K_sat=K_dry+A/B
    G_sat = G_dry # At low frequencies, Gassmann’s relations predict no change in the shear modulus between dry and saturated patches
    return K_sat,G_sat

def model(las_file, well):
    st.title('Model')
    if not las_file:
        st.warning('No file has been uploaded')
    else:
        st.write('**Curve Information**')
        
        # specify model parameters
        phi=np.linspace(0,1,100,endpoint=True) # solid volume fraction = 1-phi
        K0, G0= 32,44 # moduli of grain material
        Kw, Gw= 2.2,0 # moduli of water 
        phic = .5
        phi_new = phi/phic

        # VRH bounds
        volumes = np.vstack((1-phi,phi)).T
        M = np.array([K0,Kw])
        K_v,K_r,K_h=VRH(volumes,M)

        # Compute dry-rock moduli
        K_dry, G_dry = cripor(K0, G0, phi, phic)
        # saturate rock with water 
        # Ksat, Gsat = Gassmann(K_dry,G_dry,K0,Kw,phi)

        # porosity = (bulk density - matrix density) / (fluid density - matrix density)
        # Dbulk = bulk density
        # Dw = fluid density
        # Dmat = matrix density
        Dw = 1
        Dmat = 2.65 
        well["Phi"] = (well["DENS"]-Dmat)/(Dw-Dmat)


        # K_dry=Inverse_Gassmann(well["K"],K0,Kw,well['Phi'])
        # Ksat= Gassmann(K_dry,K0,Kw,well['Phi'])

        positive_K_dry_indices = K_dry > 4
        filtered_phi_dry = phi[positive_K_dry_indices]
        filtered_K_dry = K_dry[positive_K_dry_indices]
        print(well["Phi"])

        # positive_Ksat_indices = Ksat > 4
        # filtered_phi_sat = phi[positive_Ksat_indices]
        # filtered_Ksat = Ksat[positive_Ksat_indices]

        # Hashin-Strikmann bound 
        K_UHS,G_UHS= HS(1-phi, K0, Kw,G0,Gw, bound='upper')
        K_LHS,G_LHS= HS(1-phi, K0, Kw,G0,Gw, bound='lower')

        Kplt = well["K"]
        phi_plt = well["Phi"]
        Kplt_indices = well["Vshale"] < .5
        phi_plt = phi_plt[Kplt_indices]
        Kplt = Kplt[Kplt_indices]

        # plot
        plt.figure(figsize=(6,6))
        plt.xlabel('Porosity')
        plt.xlim(0, 1)
        plt.ylabel('Bulk modulus [GPa]')
        plt.ylim(0, 35)
        plt.title('V, R, VRH, HS bounds')
        plt.plot(phi, K_v,label='K Voigt')
        # plt.plot(phi, K_r,label='K Reuss')
        plt.plot(phi, K_r,label='K Reuss = K HS-')
        plt.plot(phi, K_h,label='K VRH')
        plt.plot(phi, K_UHS,label='K HS+')
        # plt.plot(phi, K_LHS,label='K HS-')
        plt.plot(filtered_phi_dry, filtered_K_dry,label='dry rock K')
        # plt.plot(filtered_phi_sat, filtered_Ksat,label='saturated K')

        # plt.scatter(well['Phi'], K_dry,label='dry rock K')
        # plt.scatter(well['Phi'], Ksat,s=1,label='saturated K')

        plt.scatter(phi_plt,Kplt,s=1,label="K well")

        plt.legend(loc='best')
        plt.grid(ls='--')


        # plt.savefig(path+"\\"+area+"\\"+file+"_bounds.png", dpi=150)
        # plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        



