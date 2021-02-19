import matplotlib.pyplot as plt
import random
import sys
import py21cmfast as p21c
from astropy.cosmology import WMAP9 as cosmo
import numpy as np
import time
from scipy.integrate import quad


Sigma_T = (6.6524e-25)  # Thomson scattering cross section in cm^-2
Zreion_HeII = 3 #redshift of helium reionization, for tau_e calculation
C  = (29979245800.0)  #/*  speed of light  (cm/s)  */
G=6.67259e-8 #/* cm^3 g^-1 s^-2*/
Y_He = (0.245) ##Helium fraction
m_p = (1.6726231e-24) #/* proton mass (g) */
CMperMPC = (3.086e24) #centimeters per MegaParsec

def drdz(z): ##/* comoving distance (in cm) per unit redshift */
    return (1.0+z)*C*dtdz(z)
def dtdz(z): #/* function DTDZ returns the value of dt/dz at the redshift parameter z. */
    x = np.sqrt( OMl/OMm ) * (1+z)**(-3.0/2.0)  
    dxdz = np.sqrt( OMl/OMm ) *(1+z)**(-5.0/2.0) * (-3.0/2.0)
    const1 = 2 * np.sqrt( 1 + OMm/OMl ) / (3.0 * Ho) 
    numer = dxdz * (1 + x*( x**2 + 1)**(-0.5))
    denom = x + np.sqrt(x**2 + 1)
    return (const1 * numer / denom)

class tau_e_params:  #class for stuff needed in the optical depth calculation
    def __init__(self, z, xH, length):
        self.z = z
        self.i = xH
        self.length=length

def dtau_e_dz(z_en, z, xH, length): ##calculation of optical depth per redshift
    p = tau_e_params(z,xH,length)
    if ((p.length == 0) or not(p.z)):
        return (1+z_en)*(1+z_en)*drdz(z_en)  
    else:
        if(p.z[0]< z_en):
            return (1+z_en)*(1+z_en)*drdz(z_en)
        for i in range(1, p.length):
            if(p.z[i]<z_en):
                break
        if (i==p.length):
            return 0
        xH_inter = p.xH[i-1] + (p.xH[i] - p.xH[i-1])/(p.z[i] - p.z[i-1]) * (z_en - p.z[i-1])
        xi = 1.0-xH_inter
        if(xi<0): xi=0
        if(xi>1): xi=1
        return xi*(1+z_en)*(1+z_en)*drdz(z_en)

def tau_e(zstart, zend, z_array, x_array, length): #integration of the previous function to get the optical depth
    p = tau_e_params(z_array,x_array,length)
    if (zend > Zreion_HeII and zstart < Zreion_HeII): 
        prehelium,_=quad(dtau_e_dz,Zreion_HeII, zstart, args=(z_array,x_array,length))
        posthelium,_ = quad(dtau_e_dz, zend, Zreion_HeII, args=(z_array,x_array,length))
    elif (zend > Zreion_HeII and zstart >= Zreion_HeII):
        prehelium=0
        posthelium,_=quad(dtau_e_dz, zend, zstart,args=(z_array,x_array,length) )
    else:
        posthelium=0
        prehelium,_=quad(dtau_e_dz, zend,zstart, args=(z_array,x_array,length))
    return Sigma_T*( (N_b0+He_No)*prehelium + N_b0*posthelium )


def Proj_array(z, density, velocity, xH,distance,BOX_LEN,HII_DIM,PARALLEL_APPROX=False, rotation=True):
    """
    Takes the lightcone data and compute a kSZ map for a given box
    
    Should not be accessed directly, do_kSZ calls the function itself.
    
    Parameters
    ----------
    z : float
        The starting redshift of the box
    density : array of floats of shape (HII_DIM,HII_DIM,HII_DIM)
        Density file
    velocity : array of floats of shape (HII_DIM,HII_DIM,HII_DIM)
        Velocity file
    xH : array of floats of shape (HII_DIM, HII_DIM, HII_DIM)
        xH box
    distance : float
        distance to the start of the given box in Mpc
    BOX_LEN : int
        Length of each side of the box in Mpc
    HII_DIM : int
        Size of each of the boxes
    PARALLEL_APPROX : bool, optional
        Whether to use the parallel approximation to calculate the kSZ power spectrum, default is false
    rotation : bool optional
        Whether to rotate the lightcone boxes, default is True
        
    note:
        This function is useful for rotating the boxes and for easy use with v2. 21cmFAST
    """
    
    xi=1-xH
    tx = int((HII_DIM*random.random()))  # //Put in your favorite random # generator
    ty = int((HII_DIM*random.random()))
    global Tcmb, DA_zstart, taue_arry
    DA_zstartcurrbox = distance #comoving distance to the starting redshift
    currZ = float(z)    #current redshift
#    velocity *= CMperMPC/C
    Tcmb_3d= velocity*xi*(1.0+density)   #this is used for tcmb contribution
    dtau_3d=  N_b0*(1.0+density)*xi*Sigma_T*CMperMPC*BOX_LEN/HII_DIM  #this is used for tau_e contribution


    for k in range(HII_DIM):
        inc = (k*dR + DA_zstartcurrbox) / DA_zstart  #increment for ray tracing
        dz = - dR * CMperMPC / drdz(currZ) #redshift increment for ray tracing
        currZ += dz #current redshift increment
        dtau_3d_new=dtau_3d.T[k].T*(1+currZ)**2 #tcmb and tau_e contribution with appropriate redshift dependecies
        Tcmb_3d_new=A*(1+currZ)*Tcmb_3d.T[k].T

        if (PARALLEL_APPROX):
            new_dtau=dtau_3d_new
            new_tcmb=Tcmb_3d_new
        else:  ## in ray tracing, matrices have to be adjusted
            
            new_dtau=np.concatenate((dtau_3d_new, dtau_3d_new), axis=0)
            new_dtau=np.concatenate((new_dtau, new_dtau), axis=1)
            a=np.round(np.arange(-np.shape(dtau_3d_new)[0]/2,np.shape(dtau_3d_new)[0]/2)*inc+np.shape(dtau_3d_new)[0]/2*inc)
            new_dtau=np.take(new_dtau,a.astype(int),axis=0)
            new_dtau=np.take(new_dtau,a.astype(int), axis=1)
        
            new_tcmb=np.concatenate((Tcmb_3d_new, Tcmb_3d_new), axis=0)
            new_tcmb=np.concatenate((new_tcmb, new_tcmb), axis=1)
            a=np.round(np.arange(-np.shape(Tcmb_3d_new)[0]/2,np.shape(Tcmb_3d_new)[0]/2)*inc+np.shape(Tcmb_3d_new)[0]/2*inc)
            new_tcmb=np.take(new_tcmb,a.astype(int),axis=0)
            new_tcmb=np.take(new_tcmb,a.astype(int), axis=1)
        if rotation:
            new_dtau=np.roll(new_dtau, -tx, 0) #shifting of tcmb so there is no object repetition
            new_dtau=np.roll(new_dtau, -ty, 1)
            new_tcmb=np.roll(new_tcmb, -tx, 0) #shifting of tcmb so there is no object repetition
            new_tcmb=np.roll(new_tcmb, -ty, 1)
        taue_arry+=new_dtau  #tau_e updating
        Tcmb_factor=new_tcmb*np.exp(-taue_arry) #tcmb contribution with tau_e taken in account
        Tcmb+=Tcmb_factor # tcmb contribution addition

    mean_taue_curr_z =np.mean(taue_arry)
    Tcmb=subtract_average(Tcmb)

class kSZ_output:
    def __init__(
    self,
    kSZ_box,
    taue_box,
    means=None,
    l_s=None,
    kSZ_power=None,
    cosmo_params=None
    ):
        self.kSZ_box=kSZ_box
        self.taue_box=taue_box
        self.means=means
        self.l_s=l_s
        self.kSZ_power=kSZ_power
        self.cosmo_params=cosmo_params       

def do_kSZ(lc,
             z_start = None,
    PARALLEL_APPROX=False, rotation=True, random_seed=None):
    """
    This is the main module of the program. It takes the lightcone and calculates the kSZ power spectrum.
    Parameters
    
    ---------
    lightcone : LightCone object
        LightCone object for which the kSZ power spectrum is calculated.
    z_start: float, optional
        starting redshift for kSZ calculation, default is 5
    PARALLEL_APPROX : bool, optional
        flag for parrallel approximation, Default: False
    rotation : bool, optional
        flag for rotation of boxes, if True boxes are shifted for every HII_DIM. Default: True
    
    Returns:
    ----------
    l_s : array of floats
        multiple moments sampled by the power spectrum
    P_k: array of floats
        normalized power spectrum of the power spectrum defined as: l(l+1)/2*pi *C_l [\microK^2]
    errs: array of floats:
        normalized cosmic variance errors of the power spectrum
        
        
    Notes: 
        Parallel approximation is not needed since it doesn't speed up the calculation significantly.
        do_kSZ splits the lightcone into cubes of shape (HII_DIM**3) so that the rotation is implied consistently
        and that v2. 21cmFAST can be implemented easier.
    """

    global HII_DIM, BOX_LEN
    BOX_LEN=int(lc.user_params.BOX_LEN)
    HII_DIM=int(lc.user_params.HII_DIM)
    if z_start:
        z_start=float(z_start)
    else:
        z_start=lc.lightcone_redshifts[-1]

    global START_REDSHIFT, OMm, OMl, He_No, No, OMb, N_b0, hlittle, Ho, RHOcrit_cgs, taue_arry, Tcmb
    hlittle= lc.cosmo_params.hlittle ##taking normalized Hubble parameter from the lightcone
    OMm=lc.cosmo_params.OMm ## taking matter density parameter from the lightcone
    OMb=lc.cosmo_params.OMb ##taking baryion density parameter from the lightcone
    OMl=1-OMm 
    Ho=hlittle*3.2407e-18 #/* Hubble parameter[s^-1] at z=0 */
    RHOcrit_cgs = (3.0*Ho*Ho / (8.0*np.pi*G)) #g pcm^-3 */ /* at z=0 */
    He_No = (RHOcrit_cgs*OMb*Y_He/(4.0*m_p)) #/*  current helium number density estimate */
    No  = (RHOcrit_cgs*OMb*(1-Y_He)/m_p) #current hydrogen number density estimate  (#/cm^3)  ~1.92e-7*/
    N_b0 = (No+He_No)  ##present-day baryon num density, H + He */
    Tcmb=np.zeros((HII_DIM,HII_DIM))
    mean_taue_curr_z = tau_e(0, z_start, None, None, 0)# mean optical depth
    taue_arry=np.full((HII_DIM,HII_DIM), mean_taue_curr_z)
    global A, dR
    dR=BOX_LEN/HII_DIM  #dR is cell resolution 
    redshifts=np.array(lc.lightcone_redshifts)
    red_dist=len(redshifts)
    A = N_b0*Sigma_T*BOX_LEN*CMperMPC/HII_DIM   #this is the factor to get the kSZ signal
    start = time.time()
    amount = int(red_dist/HII_DIM)  #amount of boxes with redshift dimension equal to space dimension
    
    lc_v_last=lc.velocity[:,:,amount*HII_DIM:]   #this little part takes the last part of the lightcone, pads it with appropriate value so that it can be passed further.
    lc_v_needed=np.zeros((HII_DIM,HII_DIM,HII_DIM))
    lc_v_needed[:,:,:red_dist-amount*HII_DIM]=lc_v_last
    lc_x_last=lc.xH_box[:,:,amount*HII_DIM:]
    lc_x_needed=np.ones((HII_DIM,HII_DIM,HII_DIM))
    lc_x_needed[:,:,:red_dist-amount*HII_DIM]=lc_x_last
    lc_dens_last=lc.density[:,:,amount*HII_DIM:]
    lc_dens_needed=np.zeros((HII_DIM,HII_DIM,HII_DIM))
    lc_dens_needed[:,:,:red_dist-amount*HII_DIM]=lc_dens_last
    
    density_boxes_sorted=[lc.density[:,:,i*HII_DIM:(i+1)*HII_DIM] for i in range(amount)]  #Boxes which gp into Proj_array
    density_boxes_sorted.append(lc_dens_needed)
    velocity_boxes_sorted=[lc.velocity[:,:,i*HII_DIM:(i+1)*HII_DIM] for i in range(amount)]
    velocity_boxes_sorted.append(lc_v_needed)
    xH_boxes_sorted=[lc.xH_box[:,:,i*HII_DIM:(i+1)*HII_DIM] for i in range(amount)]
    xH_boxes_sorted.append(lc_x_needed)
    redshifts_xe_sorted=[redshifts[i*HII_DIM] for i in range(amount)] #redshifts of the start of each box
    global DA_zstart
    DA_zstart = lc.lightcone_distances[0] #distance to the first slice
    distances=[lc.lightcone_distances[i*HII_DIM] for i in range(amount+1)]
    for i,z in enumerate(redshifts_xe_sorted):
        Proj_array(z,density_boxes_sorted[i], velocity_boxes_sorted[i], xH_boxes_sorted[i],distances[i], BOX_LEN,HII_DIM,PARALLEL_APPROX=PARALLEL_APPROX, rotation=rotation)


    
    l_s, P_k,errs = powpow(Tcmb*CMperMPC/C*10**6*cosmo.Tcmb0.value,     distances[0], BOX_LEN, redshifts_xe_sorted[0])

    
    return l_s,P_k,errs

def powpow(box, distance,BOX_LEN, redshifts=5):  #calculation of the power spectrum
    """
    calculation of the power spectrum of the box
    
    Parameters:
    ----------
    Box: array of shape (HII_DIM**2)
        This is the array for which power spectrum is calculated
    distance : float
        Distance to the start of the lightcone
    BOX_LEN: int
        Box length in megaparsecs
    redshifts: float, optinal
        Redshift of the start of the lightcone, default is 5
        
    Returns:
    ----------
    l_s : array of floats
        multiple moments sampled by the power spectrum
    P_k: array of floats
        normalized power spectrum of the power spectrum defined as: l(l+1)/2*pi *C_l [\microK^2]
    errs: array of floats:
        normalized cosmic variance errors of the power spectrum
    """
    dimension=box.shape[0]
    HII_DIM=dimension
    dvol = (BOX_LEN/dimension)**2
    ff=np.fft.rfft2(box)
    ff=np.fft.fftshift(ff, axes=(0,))
    ff*=dvol
    dk = 2.*np.pi/BOX_LEN
    volume =BOX_LEN**2
    K=np.zeros((int(dimension*np.sqrt(2))))
    Count=np.zeros((int(dimension*np.sqrt(2))))
    Pk=np.zeros((int(dimension*np.sqrt(2))))
    for i in range(dimension):
        for k in range(int(dimension/2+1)):
            a = 1
            if(k == 0 or k == dimension/2 + 1):
                a = 1
            m = np.sqrt(knum(HII_DIM,i)*knum(HII_DIM,i) +(k)*(k))
            n = int(np.round(m))
            K[n] += a*dk*m
            Pk[n] += a*np.abs(ff[i,k])**2 #( np.real(ff[i,k]*np.real(ff[i,k]) + ff[i,k].imag*ff[i,k].imag ))
            Count[n]+=a
    for n in range(int(dimension*np.sqrt(2)/2)):
        K[n] /= Count[n]
        Pk[n] /= (Count[n]*volume)
    l_s,Power,errs=print_stat(Pk, K, Count, dimension, redshifts, distance)
    return l_s,Power,errs
            
def print_stat(P_k,K,Count, dimension, redshifts, distance): 
    """function that calculates the cosmologically normalized power spectrum from the pure power spectrum
    """
    x = distance
    l_s=[]
    Power=[]
    err=[]

    for i in range(int(dimension*np.sqrt(2)/2)):
        coef = K[i]*K[i]/(2.*np.pi)
        l_s.append(x*K[i])
        Power.append(coef*P_k[i])
        err.append(coef*P_k[i]/np.sqrt(Count[i]/2))

    return l_s,Power,err

def subtract_average(box): #subtracts the average value of the box from the box
    return box-np.mean(box)

def knum(HII_DIM,i):
    if(i < HII_DIM/2 +1):
        return HII_DIM/2-i
    else:
        return -(HII_DIM/2 - i)

#if __name__ == '__main__':
#    do_kSZ(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])#, sys.argv[5])#, sys.argv[6])#, sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11])
