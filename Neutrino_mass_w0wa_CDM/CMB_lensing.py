import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
from scipy.interpolate import interp1d #Warren

data_dir='CMBDatafiles'


def getCkappa(ls):
    #l_vec,cl_tt, cl_ee, cl_te, cl_tt_lens, cl_ee_lens, cl_bb_lens, cl_te_lens, cl_kk=spectra()

    lC=np.loadtxt('CMBDatafiles/CAMB/cl_kk')
    #print lC
    #print lC[:,0]
    #print lC[:,1]
    l_vec=lC[:,0]
    cl_kk=lC[:,1]
    kappaSpline=interp1d(l_vec, cl_kk, bounds_error=False,fill_value=0.0)
    return kappaSpline(ls)

def getN0kappa(ls, expt, spec):  #N0 reconstruction noise
    LN=np.loadtxt(data_dir+'/N0/'+expt+'_N_'+spec)
    L=LN[0,:]
    N0pp=LN[1,:]#N0dd=LN[1,:]
    N0kk=L**4/4*N0pp#N0kk=L**4*N0dd
    kappaNoiseSpline=spline1d(L, N0kk)# bounds_error=False,fill_value=1.)
    if spec=='tt':
        print 'WARNING: noise spline after l=300 is dodgy for tt - need to sort out datafile'
    return kappaNoiseSpline(ls)



if __name__=='__main__':
    output_dir = 'Warren_new_plots_output/'
    ls=np.arange(0,2000)
    Ckk=getCkappa(ls)
    N0kk=getN0kappa(ls, 'ref','eb')
    l_fac=1#ls*(ls+1)/(2*np.pi)
    from pylab import *
    rc('axes', linewidth=1.2)

    plt.plot(ls, l_fac*Ckk,linewidth=2.)
    plt.plot(ls, l_fac*N0kk,'--',linewidth=2.)
    plt.ylim(1e-9, 1e-6)
    plt.xlim(2,2000)
    plt.xscale('log')
    plt.yscale('log')
    #plt.xticks(fontsize=10)
    #plt.yticks(fontsize=10)
    #plt.title('Cdd and N0dd')
    plt.ylabel(r'$C_\ell^{\kappa \kappa}$',fontsize=15)
    plt.xlabel(r'$\ell$',fontsize=15)
    plt.savefig(output_dir+'CMB_convergence_spectra.png',dpi=300)
    plt.show()

    delta_ls = ls[1]-ls[0]
    fsky=0.4
    Nmodes= (2.*ls+1.)*delta_ls*fsky
    SNR = np.sqrt(Nmodes)*(Ckk/(Ckk+N0kk))
    cumSNR = np.cumsum(SNR**2.)
    plt.plot(ls,np.sqrt(cumSNR))
    plt.ylabel(r'SNR$_{\kappa \kappa}$',fontsize=15)
    plt.xlabel(r'$\ell$',fontsize=15)
    plt.savefig(output_dir+'CMB_convergence_SNR.png',dpi=300)
    plt.show()

    fig, ax = plt.subplots(2, 1, sharex=True)
 # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    ax[0].plot(ls, l_fac*Ckk,linewidth=2.)
    ax[0].plot(ls, l_fac*N0kk,'--',linewidth=2.)
    ax[0].set_ylim(1e-9, 1e-6)
    ax[0].set_xlim(2,2000)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_ylabel(r'$C_\ell^{\kappa \kappa}$',fontsize=17)

    ax[0].set_xlabel(r'$\ell$',fontsize=17)
    ax[1].plot(ls,np.sqrt(cumSNR))
    ax[1].set_ylabel(r'SNR$_{\kappa \kappa}$',fontsize=17)
    ax[1].set_xlabel(r'$\ell$',fontsize=15)
    ax[0].tick_params(axis='both',labelsize=15)
    ax[1].tick_params(axis='both',labelsize=15)
    plt.show()
