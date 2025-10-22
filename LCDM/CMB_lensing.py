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


#I am using camb for Python now instead of Simon's datafiles
#def spectra():
#    lmax=9999       #what CAMB stuff goes up to (2 to 10000 ignoring 10100)
#    lmax2=105000     #a bit over what we need for ACT resolution (33000ish)
#    data_scal=np.loadtxt(data_dir+'/CAMB/new_scalCls.dat')
#    data_lens=np.loadtxt(data_dir+'/CAMB/new_lensedCls.dat')
#
#    data_scal=data_scal[0:lmax,:]
#    data_lens=data_lens[0:lmax,:]
#    pp_factor=7.4311e+12  # factor that normalises large-scale structure relative to cmb, (10^6 *Tcmb)^2
#    l_vec=data_scal[:,0]
#    l_arr_sq = l_vec*l_vec
#    rescale = l_vec * (l_vec + 1.0)/(2*np.pi)   #Jet commented out division by 2pi, I put it back after reading camb readme
#
#    cl_tt=(data_scal[:,1]/(rescale))
#    cl_ee=(data_scal[:,2]/(rescale))
#    cl_te=(data_scal[:,3]/(rescale))
#
#    cl_tt_lens=data_lens[:,1]/(rescale)
#    cl_ee_lens=data_lens[:,2]/(rescale)
#    cl_bb_lens=data_lens[:,3]/(rescale)
#    cl_te_lens=data_lens[:,4]/(rescale)
#
#
#
#    cl_dd=(data_scal[:,4]/l_arr_sq)/pp_factor
#    cl_pp=cl_dd/l_arr_sq   #made this agree with CAMB readme by replacing rescale with l_arr_sq
#    cl_kk=(rescale*2*np.pi)**2*cl_pp/4 #after putting back the /2pi in rescale, I cancelled it out here by multiplying by 2pi
#
#
#
#
#
#    #making longer so that when we need the spectrum for large l we can give a sensible answer
#    l_vec_extra=np.arange(l_vec[l_vec.size-1]+1,l_vec[l_vec.size-1]+1+(lmax2-lmax), 1) #need to go out further for higher resolution
#    l_vec_long=np.concatenate((l_vec,l_vec_extra))
#
#    cl_tt_lens_extra=(l_vec_extra**(-6))
#    cl_tt_lens_extra=cl_tt_lens_extra*cl_tt_lens[l_vec.size-1]/cl_tt_lens_extra[0]
#    cl_tt_lens_long=np.concatenate((cl_tt_lens,cl_tt_lens_extra))
#
#    cl_tt_extra=(l_vec_extra**(-5))
#    cl_tt_extra=cl_tt_extra*cl_tt[l_vec.size-1]/cl_tt_extra[0]
#    cl_tt_long=np.concatenate((cl_tt,cl_tt_extra))
#
#    cl_ee_lens_extra=(l_vec_extra**(-6))
#    cl_ee_lens_extra=cl_ee_lens_extra*cl_ee_lens[l_vec.size-1]/cl_ee_lens_extra[0]
#    cl_ee_lens_long=np.concatenate((cl_ee_lens,cl_ee_lens_extra))
#
#    cl_ee_extra=(l_vec_extra**(-5.5))
#    cl_ee_extra=cl_ee_extra*cl_ee[l_vec.size-1]/cl_ee_extra[0]
#    cl_ee_long=np.concatenate((cl_ee,cl_ee_extra))
#
#    cl_te_lens_extra=(l_vec_extra**(-6))
#    cl_te_lens_extra=cl_te_lens_extra*cl_te_lens[l_vec.size-1]/cl_te_lens_extra[0]
#    cl_te_lens_long=np.concatenate((cl_te_lens,cl_te_lens_extra))
#
#    cl_te_extra=(l_vec_extra**(-5.5))
#    cl_te_extra=cl_te_extra*cl_te[l_vec.size-1]/cl_te_extra[0]
#    cl_te_long=np.concatenate((cl_te,cl_te_extra))
#
#    cl_bb_lens_extra=(l_vec_extra**(-6))
#    cl_bb_lens_extra=cl_bb_lens_extra*cl_bb_lens[l_vec.size-1]/cl_bb_lens_extra[0]
#    cl_bb_lens_long=np.concatenate((cl_bb_lens,cl_bb_lens_extra))
#
#    cl_kk_extra=(l_vec_extra**(-2.8))
#    cl_kk_extra=cl_kk_extra*cl_tt[l_vec.size-1]/cl_kk_extra[0]
#    cl_kk_long=np.concatenate((cl_kk,cl_kk_extra))
#
#    return(l_vec_long,cl_tt_long, cl_ee_long, cl_te_long, cl_tt_lens_long, cl_ee_lens_long, cl_bb_lens_long, cl_te_lens_long, cl_kk_long)
#


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
