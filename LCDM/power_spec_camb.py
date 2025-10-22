# from http://camb.readthedocs.io/en/latest/CAMBdemo.html
# need to have pycamb installed to run this and to get matter power spectrum and lensing convergence power spectrum datafiles if you change cosmological_parameters
#still to install (prob reason for errors): six (compatibility between python 2 and 3) and mock (for testing)
import cosmological_parameters
import sys, platform, os
from matplotlib import pyplot as plt
import numpy as np
import cosmolopy
#import constants
import camb
from camb import model, initialpower
import time
start_time = time.time()

show_plots=True

zs=np.linspace(0,1,1)

#Now get matter power spectra at various z
pars = camb.CAMBparams()
pars.set_cosmology(H0=cosmological_parameters.H0, ombh2=cosmological_parameters.ombh2, omch2=cosmological_parameters.omch2)
pars.set_dark_energy() #re-set defaults
pars.InitPower.set_params(ns=cosmological_parameters.ns)
#lensing
pars.set_for_lmax(5000, lens_potential_accuracy=0)
results = camb.get_results(pars)
cl = results.get_lens_potential_cls(lmax=5000)

plt.loglog(np.arange(5001), cl[:,0])
plt.ylabel('$[L(L+1)]^2C_L^{\phi\phi}/2\pi$')
plt.xlabel('$L$')
#plt.xlim([2,2000])
plt.show()

cl_kap=cl[:,0]*2*np.pi/4

plt.loglog(np.arange(5001), cl_kap)
plt.xlabel('$L$')
plt.ylabel('$C_L^{\kappa\kappa}$')
plt.ylim((1e-9, 1e-6))
plt.show()

lC=np.zeros((5001, 2))
lC[:,0]=np.arange(5001)
lC[:,1]=cl_kap
#np.savetxt('CMBDatafiles/CAMB/cl_kk', lC)



#Not non-linear corrections couples to smaller scales than you want
pars.set_matter_power(redshifts=zs, kmax=18.0)

#Linear spectra
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)

#matter
kh, z, pk = results.get_matter_power_spectrum(minkh=3e-5, maxkh=18, npoints = 10000)
print(pars)
#Non-Linear spectra (Halofit)
pars.NonLinear = model.NonLinear_both
results.calc_power_spectra(pars)
kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=3e-5, maxkh=18, npoints = 10000)



print("--- %s seconds ---" % (time.time() - start_time))

if show_plots:
    for i, redshift in enumerate(z):
        plt.loglog(kh, pk[i,:], label='z='+str(redshift))
        plt.loglog(kh_nonlin, pk_nonlin[i,:], color='r', ls = ':')
    plt.xlabel(r'$k [h Mpc^{-1}]$');
    plt.ylabel(r'$P(k) [h^{-3} Mpc3]$')
    plt.legend(loc='lower left');
    plt.title('Linear and nonlinear matter power spectra');

    print("--- %s seconds ---" % (time.time() - start_time))

    plt.show()

kP=np.zeros((kh_nonlin.size, 2))
kP[:,0]=kh_nonlin
kP[:,1]=pk_nonlin[0,:]
#np.savetxt('Datafiles/P_k_nonlin_camb_z0.dat', kP)

kP=np.zeros((kh.size, 2))
kP[:,0]=kh
kP[:,1]=pk[0,:]
#np.savetxt('Datafiles/P_k_lin_camb_z0.dat', kP)

z = np.linspace(0,4,100)
DA = results.angular_diameter_distance(z)
Hz = results.h_of_z(z)
plt.plot(z, DA)
plt.xlabel('$z$')
plt.ylabel(r'$D_A /\rm{Mpc}$')
plt.title('Angular diameter distance')
#plt.ylim([0,2000]);
plt.show()
