import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate


import cosmological_parameters

import sys
sys.path.append('cosmolopy')
import cosmolopy.distance as cd
cosmo = {'omega_M_0':cosmological_parameters.om_m, 'omega_lambda_0':cosmological_parameters.om_L, 'omega_k_0':0.0, 'h':cosmological_parameters.h} #check latest, defining cosmology for cosmolopy distance measures


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# lensing kernel
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def get_lensing_kernel_real_sp(chi_K, show_plots=False):


    zW=np.loadtxt('Datafiles/lensing kernel.dat')
    z=zW[:,0]
    W_kap=zW[:,1]

    W_kap_z_spl=interp1d(z,W_kap,bounds_error=False,fill_value=0.0)

    zD=np.loadtxt('Datafiles/growth_function_D.dat')
    z2=zD[:,0]
    D=zD[:,1]

    D_spl=interp1d(z2,D,bounds_error=False,fill_value=0.0)

    chi=cd.comoving_distance(z, **cosmo)    #Mpc

    D_H=cd.hubble_distance_z(z, **cosmo)


    K=(1/D_H)*W_kap*D_spl(z)#/chi**2
    #K=(1/D_H)*W_kap#*D_spl(z)

    K_chi_spl=interp1d(chi, K,bounds_error=False,fill_value=0.0)

    if show_plots:
        plt.plot(chi, W_kap)
        plt.xlabel(r'$\chi$ [Mpc]')
        plt.ylabel('W')
        plt.show()

        plt.plot(chi, K)
        plt.xlabel(r'$\chi$  [Mpc]')
        plt.ylabel('K')
        plt.title(r'Real space HI kernel')
        plt.show()

    return K_chi_spl(chi_K)


def real_space_kernel_plots():
    n=8192*16
    chi_lss = cd.comoving_distance(1090, **cosmo)
    chi_full=np.linspace(0,chi_lss,n)
    chi_bin1 = np.linspace(2798.,3005.,n)
    chi_bin2 = np.linspace(3005.,3564.,n)
    chi_bin3 = np.linspace(3564.,4523.,n)
    chi_bin4 = np.linspace(4523.,6032.,n)
    kernel_bin1 = interp1d(chi_bin1,get_lensing_kernel_real_sp(chi_bin1),bounds_error=False,fill_value=0.0)(chi_full)
    kernel_bin2 = interp1d(chi_bin2,get_lensing_kernel_real_sp(chi_bin2),bounds_error=False,fill_value=0.0)(chi_full)
    kernel_bin3 = interp1d(chi_bin3,get_lensing_kernel_real_sp(chi_bin3),bounds_error=False,fill_value=0.0)(chi_full)
    kernel_bin4 = interp1d(chi_bin4,get_lensing_kernel_real_sp(chi_bin4),bounds_error=False,fill_value=0.0)(chi_full)
    plt.plot(chi_full,get_lensing_kernel_real_sp(chi_full),label=r'full $\chi_\parallel$ range')
    plt.plot(chi_full,kernel_bin4,label=r'$z_{c,i} = 1.95$')
    plt.plot(chi_full,kernel_bin3,label=r'$z_{c,i} = 1.27$')
    plt.plot(chi_full,kernel_bin2,label=r'$z_{c,i} = 0.95$')
    plt.plot(chi_full,kernel_bin1,label=r'$z_{c,i} = 0.81$')
    plt.ylim(0)
    plt.xlabel(r'$\chi_\parallel$ $[Mpc]$',fontsize=15)
    plt.ylabel(r'$K(\chi_\parallel)$ $[Mpc^{-1}]$',fontsize=15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #plt.xticks(fontsize=15)
    #plt.yticks(fontsize=15)
    plt.legend(loc='upper right',fontsize=15)
    plt.savefig('Warren_new_plots_output/'+'CMB_kernel_real_space.png',dpi=300)
    plt.show()

def get_lensing_kernel_fourier_sp_bin(k_par_in, chi_min, chi_max,show_plots=False):

    n=8192*16
    delta_chi = chi_max - chi_min
    d=delta_chi/n
    chi_lss = cd.comoving_distance(1090, **cosmo)


    chi_bin=np.linspace(chi_min, chi_max, n)
    chi_full=np.linspace(0,chi_lss,n)


    K_chi=get_lensing_kernel_real_sp(chi_bin, show_plots)
    K_full = interp1d(chi_bin,K_chi,bounds_error=False,fill_value=0.0)(chi_full)

    K_k_par=np.fft.fft(K_full)*d
    k_par=np.fft.fftfreq(n,d)
    K_spl=interp1d(k_par[:n/2], np.abs(K_k_par[:n/2])*np.cos(k_par[:n/2]*delta_chi*2*np.pi),bounds_error=False,fill_value=0.0 )

    K_final=K_spl(k_par_in)

    if show_plots:
        plt.plot(k_par[:n/2], K_k_par[:n/2]*np.cos(k_par[:n/2]*delta_chi*2*np.pi))
        plt.show()
        plt.plot(chi_full,K_full)
        plt.show()
    return K_final


def get_lensing_kernel_fourier_sp(k_par_in, show_plots=False): #K_k_par

    chi_max=cd.comoving_distance(1090, **cosmo)
    n=8192*16

    d=chi_max/n

    chi_f=np.linspace(0, chi_max, n)

    K_chi=get_lensing_kernel_real_sp(chi_f, show_plots)


    K_k_par=np.fft.fft(K_chi*d)
    k_par=np.fft.fftfreq(n,d)
    K_spl=interp1d(k_par[:n/2],  np.abs(K_k_par[:n/2])*np.cos(k_par[:n/2]*chi_max*2*np.pi),bounds_error=False,fill_value=0.0 )

    K_final=K_spl(k_par_in)

    if show_plots:
        plt.loglog(k_par[:n/2], np.abs(K_k_par[:n/2])*np.cos(k_par[:n/2]*chi_max*2*np.pi))
        plt.xlabel(r'$k_\parallel$')
        plt.ylabel('|K|')
        plt.savefig('TestingPlots/Magnitude of HS 21cm kernel')
        plt.title(r'Magnitude of HI kernel in $k_\parallel$ space')
        plt.show()

    return K_final

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# main
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__=='__main__':
    from pylab import *
    rc('axes', linewidth=1.2)
    #real_space_kernel_plots()
    k_par_in=np.logspace(-5, -1, 100)
    kernel=get_lensing_kernel_fourier_sp(k_par_in, True)
    #np.savetxt('Full_HIRAX_Lensing_Kernel_kpar.dat',(k_par_in, kernel))
    np.savetxt('Full_HIRAX_Lensing_Kernel_kpar_no_D.dat',(k_par_in, kernel))
    raise KeyboardInterrupt

    plt.plot(k_par_in, kernel, label='Full kernel')
    plt.plot(k_par_in,get_lensing_kernel_fourier_sp_bin(k_par_in,4523.,6032.,show_plots=False),label='$z_i=1.95$')
    plt.plot(k_par_in,get_lensing_kernel_fourier_sp_bin(k_par_in,3564.,4523.),label='$z_{c,i}=1.27$')
    plt.plot(k_par_in,get_lensing_kernel_fourier_sp_bin(k_par_in,3005.,3564.),label='$z_{c,i}=0.95$')
    plt.plot(k_par_in,get_lensing_kernel_fourier_sp_bin(k_par_in,2798.,3005.),label='$z_{c,i}=0.81$')
    plt.plot(k_par_in,get_lensing_kernel_fourier_sp_bin(k_par_in,2798.,6032.,show_plots=False),label='Full HIRAX range')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$k_\parallel$ $[Mpc^{-1}]$',fontsize=15)
    plt.ylabel(r'$K(k_\parallel)$',fontsize=15)
    #plt.xticks(fontsize=25)
    #plt.yticks(fontsize=25)
    #plt.title('Lensing Kernel in Harmonic Space')
    plt.legend(loc='lower left',ncol=3,fontsize=11)
    plt.ylim(1e-6,1e0)
    plt.savefig('Warren_new_plots_output/'+'CMB_kernel_fourier_space.png',dpi=300)
    plt.show()
    plt.xlim([0,0.5])
