##########################################################################################################
# In this module we define the derivative functions of the HI, CMB lensing and bispectrum fields that
# are required for the Fisher matrix calculation. We then also define functions that compute the Fisher matrices for
# each of the three probes in three cases, namely the LCDM case, the w0waCDM case and finally
# Nuetrino mass-w0wa CDM case.
#########################################################################################################

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,                                                  mark_inset)
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
from matplotlib import cm
from matplotlib import colors
from scipy.interpolate import interp1d
import os, copy
import scipy
import pylab
import matplotlib.patches as mpatches
import matplotlib
import CMB_lensing
import HI_experiments
from scipy.stats import chi2
import settings
import binning
import lensing_kernel
import cosmological_parameters
import power_spec_functions
import sys
import cosmolopy.distance as cd
import cosmolopy.density as density
import Full21cmblens_primary_functions as Fpm

sys.path.append('cosmolopy')
cosmo = {'omega_M_0':cosmological_parameters.om_m, 'omega_lambda_0':cosmological_parameters.om_L, 'omega_k_0':0.0, 'h':cosmological_parameters.h}

output_dir = 'Output/'

if not os.path.exists(output_dir):
		os.makedirs(output_dir)

SAVEFIG = settings.SAVEFIG
SHOWFIG = settings.SHOWFIG

expt=HI_experiments.getHIExptObject(settings.expt_name, settings.expt_mode)
fsky = expt.getfsky()
Sarea_sr = 4.*np.pi*fsky


def EOS_parameters_derivs():

    ''' Based on BAO_Fisher forecast code by Phil Bull et al. arXiv 1405.1452,
    gives the derivatives of distance scale parameters alpha_perp and alpha_parallel
    in terms of the implicit cosmological parameters Om_k, Om_DE, w0, wa, g, gamma '''

    C = 3e5*3.24e-23
    w0 = cosmological_parameters.w0; wa = cosmological_parameters.wa
    om = cosmological_parameters.om_m; ol = cosmological_parameters.om_L
    ok = 1. - om - ol

    # Omega_DE(a) and E(a) functions
    omegaDE = lambda a: ol * np.exp(3.*wa*(a - 1.)) / a**(3.*(1. + w0 + wa))
    E = lambda a: np.sqrt( om * a**(-3.) + ok * a**(-2.) + omegaDE(a) )

    # Derivatives of E(z) w.r.t. parameters
    #dE_omegaM = lambda a: 0.5 * a**(-3.) / E(a)
    if np.abs(ok) < 1e-7: # Effectively zero
        dE_omegak = lambda a: 0.5 * a**(-2.) / E(a)
    else:
        dE_omegak = lambda a: 0.5 * a**(-2.) / E(a) * (1. - 1./a)
    dE_omegaM = lambda a: 0.5 * a**(-3.) / E(a)
    dE_omegaDE = lambda a: 0.5 / E(a) * (1. - 1./a**3.)
    dE_w0 = lambda a: -1.5 * omegaDE(a) * np.log(a) / E(a)
    dE_wa = lambda a: -1.5 * omegaDE(a) * (np.log(a) + 1. - a) / E(a)


    # Bundle functions into list (for performing repetitive operations with them)
    fns = [dE_omegak, dE_omegaDE, dE_w0, dE_wa]
    #HH, rr, DD, ff = cosmo_fns

    aa = np.linspace(1., 1e-4, 500)
    zz = 1./aa - 1.
    EE = E(aa); fz = power_spec_functions.get_growth_factor_f(aa)#ff(aa)
    gamma = cosmological_parameters.gamma; H0 = 100. * cosmological_parameters.h; h = cosmological_parameters.h

    # Derivatives of apar w.r.t. parameters
    derivs_apar = [f(aa)/EE for f in fns]


    # Derivatives of f(z) w.r.t. parameters
    f_fac = -gamma * fz / EE
    df_domegak  = f_fac * (EE/om + dE_omegak(aa))
    df_domegaDE = f_fac * (EE/om + dE_omegaDE(aa))
    df_w0 = f_fac * dE_w0(aa)
    df_wa = f_fac * dE_wa(aa)
    df_dh = np.zeros(aa.shape)
    df_dgamma = fz * np.log(density.omega_M_z(zz, **cosmo))
    derivs_f = [df_domegak, df_domegaDE, df_w0, df_wa, df_dh, df_dgamma]
    #plt.plot(aa,derivs_f[2],label='df_dw0')
    #plt.plot(aa,derivs_f[3],label='df_dwa')

    # Calculate comoving distance (including curvature)
    r_c = scipy.integrate.cumtrapz(1./(aa**2. * EE), aa)
    r_c = np.concatenate(([0.], r_c))
    if ok > 0.:
        r = C/(H0*np.sqrt(ok)) * np.sinh(r_c * np.sqrt(ok))
    elif ok < 0.:
        r = C/(H0*np.sqrt(-ok)) * np.sin(r_c * np.sqrt(-ok))
    else:
        r = C/H0 * r_c

    # Perform integrals needed to calculate derivs. of aperp
    derivs_aperp = [(C/H0)/r[1:] * scipy.integrate.cumtrapz(f(aa)/(aa * EE)**2., aa)
                        for f in fns]

    # Add initial values (to deal with 1/(r=0) at origin)
    inivals = [0.5, 0.0, 0., 0.]
    derivs_aperp = [ np.concatenate(([inivals[i]], derivs_aperp[i]))
                     for i in range(len(derivs_aperp)) ]

    # Add (h, gamma) derivs to aperp,apar
    derivs_aperp += [np.ones(aa.shape)/h, np.zeros(aa.shape)]
    derivs_apar  += [np.ones(aa.shape)/h, np.zeros(aa.shape)]

    # Construct interpolation functions
    interp_f     = [scipy.interpolate.interp1d(aa[::-1], d[::-1],
                    kind='linear', bounds_error=False) for d in derivs_f]
    interp_apar  = [scipy.interpolate.interp1d(aa[::-1], d[::-1],
                    kind='linear', bounds_error=False) for d in derivs_apar]
    interp_aperp = [scipy.interpolate.interp1d(aa[::-1], d[::-1],
                    kind='linear', bounds_error=False) for d in derivs_aperp]

    return [interp_f, interp_aperp, interp_apar]


def expand_fisher_matrix(zbin_prop, derivs, F, names, exclude=[]):
	# Based on BAO_Fisher code from Phil Bull et al. arXiv 1405.1452


    """
    Transform Fisher matrix to with (f, aperp, apar) parameters into one with
    dark energy EOS parameters (Omega_k, Omega_DE, w0, wa, h, gamma) instead.

    Parameters
    ----------

    z : float
        Central redshift of the survey.

    derivs : 2D list of interp. fns.
        Array of interpolation functions used to evaluate derivatives needed to
        transform to new parameter set. Precompute this using the
         function.

    F : array_like
        Fisher matrix for the old parameters.

    names : list
        List of names of the parameters in the current Fisher matrix.

    exclude : list, optional
        Prevent a subset of the functions [f, aperp, apar] from being converted
        to EOS parameters. e.g. exclude = [1,] will prevent aperp from
        contributing to the EOS parameter constraints.

    Returns
    -------

    Fnew : array_like
        Fisher matrix for the new parameters.

    paramnames : list, optional
        Names parameters in the expanded Fisher matrix.
    """
    z_A = zbin_prop.z_A
    a = 1. / (1. + z_A)

    # Define mapping between old and new Fisher matrices (including expanded P(k) terms)
    old = copy.deepcopy(names)
    Nold = len(old)
    oldidxs = [old.index(p) for p in ['f', 'aperp', 'apar']]

    # Insert new parameters immediately after 'apar'
    new_params = [r'$\Omega_k$', r'$\Omega_\Lambda$', 'w0', 'wa', 'h', r'$\gamma$']
    new = old[:old.index('apar')+1]
    new += new_params
    new += old[old.index('apar')+1:]
    newidxs = [new.index(p) for p in new_params]
    Nnew = len(new)

    # Construct extension operator, d(f,aperp,par)/d(beta)
    S = np.zeros((Nold, Nnew))
    for i in range(Nold):
      for j in range(Nnew):
        # Check if this is one of the indices that is being replaced
        if i in oldidxs and j in newidxs:
            # Old parameter is being replaced
            ii = oldidxs.index(i) # newidxs
            jj = newidxs.index(j)
            if ii not in exclude:
                S[i,j] = derivs[ii][jj](a)
        else:
            if old[i] == new[j]: S[i,j] = 1.

    # Multiply old Fisher matrix by extension operator to get new Fisher matrix
    Fnew = np.dot(S.T, np.dot(F, S))
    Omega_DE = cosmological_parameters.om_L
    Omega_k = cosmological_parameters.om_k
    w0 = cosmological_parameters.w0
    wa = cosmological_parameters.wa
    h = cosmological_parameters.h
    gamma = cosmological_parameters.gamma

    Fid = np.array([ Omega_k,Omega_DE, w0, wa, h, gamma])
    return  new, Fid, Fnew


################################################################################################
# Below are all LCDM derivatives and Fisher Matrices
################################################################################################

def Cl21_Distance_derivs_LCDM(zbin_prop,ell_arr, y_arr,fid_params, bias_var=1., gamma_var=1.):

   ''' Compute the derivatives of the HI signal used in Fisher matrix calculation '''

   n_ell = ell_arr.size ; n_y = y_arr.size
   z_A = zbin_prop.z_A
   chi_A = zbin_prop.chi_A ; nutilde_A = zbin_prop.nutilde_A ; rnu_A = zbin_prop.rnu_A
   kperp_arr = ell_arr/chi_A; kpar_arr = y_arr/rnu_A
   apar = cosmological_parameters.apar ; aperp = cosmological_parameters.aperp

   ktot_2d_arr = np.sqrt(np.outer(ell_arr,np.ones(n_y))**2/chi_A**2 + np.outer(np.ones(n_ell),y_arr)**2/rnu_A**2)
   kpar_2d_arr = np.outer(np.ones(n_ell),kpar_arr)
   kperp_2d_arr = np.outer(kperp_arr,np.ones(n_y))
   mu_k_2d_arr = kpar_2d_arr/ktot_2d_arr
   fbao_2d_arr = power_spec_functions.get_fbao(ktot_2d_arr)
   alpha_fnl_2d_arr = 3e5**2.*2.*ktot_2d_arr**2.* power_spec_functions.get_transfer_function(ktot_2d_arr)*power_spec_functions.get_growth_function_D(z_A)\
    				  /(3. * cosmological_parameters.om_m * (cosmological_parameters.H0)**2.)
   beta_fnl = 2.*cosmological_parameters.delta_c_fnl* ( power_spec_functions.get_HI_bias(z_A,bias_var) - 1.)


   F_bias_rsd_2d_arr = power_spec_functions.get_HI_bias(z_A, bias_var)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2
   F_bias_rsd_sq_2d_arr = F_bias_rsd_2d_arr**2
   F_bias_rsd_fnl_2d_arr = power_spec_functions.get_HI_bias(z_A, bias_var)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2 + \
   cosmological_parameters.f_nl * beta_fnl / alpha_fnl_2d_arr

   P_21_tot_2d_arr = F_bias_rsd_sq_2d_arr *power_spec_functions.get_P_m_0(ktot_2d_arr) * \
	power_spec_functions.get_mean_temp(z_A)**2 * power_spec_functions.get_growth_function_D(z_A)**2  # mK^2
   P_21_tot_2d_arr_fnl =  F_bias_rsd_fnl_2d_arr*power_spec_functions.get_P_m_0(ktot_2d_arr) * \
 	power_spec_functions.get_mean_temp(z_A)**2 * power_spec_functions.get_growth_function_D(z_A)**2

   Cl_21_auto_ell_y_2d_arr = P_21_tot_2d_arr / (chi_A**2 * rnu_A)
   Cl_21_auto_ell_y_2d_arr_fnl = P_21_tot_2d_arr_fnl / (chi_A**2 * rnu_A)

   f_zA = power_spec_functions.get_growth_factor_f(z_A, bias_var)
   drsd_du2 = 2.*f_zA/ F_bias_rsd_2d_arr

   h=cosmological_parameters.h
   kP=np.loadtxt('Datafiles/P_k_nonlin_camb_z0.dat')  # goes up to k=18/Mpc
   P_spl_k=interp1d(kP[:,0]/h, kP[:,1]*h**3,bounds_error=False,fill_value=0.0)


   dlogpk_dk = Fpm.logpk_derivative(P_spl_k, ktot_2d_arr) # Numerical deriv.
   daperp_u2 = -2. * (kperp_2d_arr/kpar_2d_arr * aperp/apar * mu_k_2d_arr**2.)**2. / aperp
   dapar_u2 =   2. * (kperp_2d_arr/kpar_2d_arr* aperp/apar * mu_k_2d_arr**2.)**2. / apar
   daperp_k = (aperp*kperp_2d_arr)**2. / (ktot_2d_arr*aperp)
   dapar_k  = (apar*kpar_2d_arr)**2. / (ktot_2d_arr*apar)

   deriv_aperp = ( (2./aperp) + drsd_du2 * daperp_u2 \
                       + (dlogpk_dk)*daperp_k ) * Cl_21_auto_ell_y_2d_arr
   deriv_apar =  ( (1./apar)  + drsd_du2 * dapar_u2 \
                       + (dlogpk_dk)*dapar_k  ) * Cl_21_auto_ell_y_2d_arr

   deriv_f = 2.*mu_k_2d_arr**2./ F_bias_rsd_2d_arr *  Cl_21_auto_ell_y_2d_arr

   Abao_zA,sig8, b1_zA, b2_zA,  f_zA, aperp, apar= fid_params
   sig8_zA = sig8*power_spec_functions.get_growth_function_D(z_A)

   dC_ell_y_dAbao = Cl_21_auto_ell_y_2d_arr*fbao_2d_arr/(1.+fbao_2d_arr)

   sig8_deriv_kernel = 2.0/(sig8_zA*power_spec_functions.get_growth_function_D(z_A))
   dC_ell_y_dsig8_zA = Cl_21_auto_ell_y_2d_arr * sig8_deriv_kernel

   b2_deriv_kernel = 0.0
   dC_ell_y_db2_zA = Cl_21_auto_ell_y_2d_arr * b2_deriv_kernel

   b1_deriv_kernel = 2.0/F_bias_rsd_2d_arr
   dC_ell_y_db1_zA = Cl_21_auto_ell_y_2d_arr*b1_deriv_kernel

   fnl_deriv_kernel = 2.0*beta_fnl/(alpha_fnl_2d_arr*F_bias_rsd_fnl_2d_arr)
   dC_ell_y_dfnl_zA = Cl_21_auto_ell_y_2d_arr*fnl_deriv_kernel

   dC_dns = np.log(ktot_2d_arr/0.05)*Cl_21_auto_ell_y_2d_arr


   return  dC_ell_y_dAbao, dC_ell_y_dsig8_zA, dC_ell_y_db1_zA, dC_dns, deriv_f, deriv_aperp, deriv_apar



def Cl_kappa_distance_derivs_LCDM(ell_arr, y_arr, zbin_prop, fid_params, bias_var=1., gamma_var=1.): # 21cm angular PS at z vs ell,y

    ''' Compute the derivatives of the CMB convergence field use in Fisher matrix calculation '''

    n_ell = ell_arr.size ; n_y = y_arr.size
    z_A = zbin_prop.z_A
    chi_A = zbin_prop.chi_A ; nutilde_A = zbin_prop.nutilde_A ; rnu_A = zbin_prop.rnu_A
    chi_max = zbin_prop.chi_A_min; chi_min = zbin_prop.chi_A_max
    kperp_arr = ell_arr/chi_A; kpar_arr = y_arr/rnu_A
    apar = cosmological_parameters.apar ; aperp = cosmological_parameters.aperp


    ktot_2d_arr = np.sqrt(np.outer(ell_arr,np.ones(n_y))**2/chi_A**2 + np.outer(np.ones(n_ell),y_arr)**2/rnu_A**2)
    kpar_2d_arr = np.outer(np.ones(n_ell),kpar_arr)
    kperp_2d_arr = np.outer(kperp_arr,np.ones(n_y))
    fbao_2d_arr = power_spec_functions.get_fbao(ktot_2d_arr)


    Cl_kappa_auto_ell_y_2d_arr = Fpm.CMB_convergence_power(kperp_2d_arr*chi_A)


    f_zA = power_spec_functions.get_growth_factor_f(z_A, bias_var)

    h=cosmological_parameters.h
    kP=np.loadtxt('Datafiles/P_k_nonlin_camb_z0.dat')  # goes up to k=18/Mpc
    P_spl_k=interp1d(kP[:,0]/h, kP[:,1]*h**3,bounds_error=False,fill_value=0.0)


    dlogpk_dk = Fpm.logpk_derivative(P_spl_k, ktot_2d_arr) # Numerical deriv.


    W_kappa_over_chi_sq=lensing_kernel.get_lensing_kernel_real_sp(chi_A)**2.*power_spec_functions.get_growth_function_D(z_A)**2. / chi_A**2.


    sig8, ns, Om_m, h  = fid_params
    sig8_zA = sig8


    sig8_deriv_kernel = 2.0/(sig8_zA)
    dC_ell_y_dsig8_zA = Cl_kappa_auto_ell_y_2d_arr * sig8_deriv_kernel

    dC_dns = 0.0

    dC_ell_y_dOm_m = -2.0*Cl_kappa_auto_ell_y_2d_arr/(Om_m)

    dC_ell_y_dh = 4.0*Cl_kappa_auto_ell_y_2d_arr/(h)


    return  dC_ell_y_dsig8_zA, dC_dns, dC_ell_y_dOm_m, dC_ell_y_dh



def bispec_distance_derivs_LCDM(ell_coarse_arr, y_coarse_arr, zbin_prop, fid_params, bias_var=1., gamma_var=1.):

    ''' Compute the derivatives of the bispectrum used in the Fisher matrix calculation '''

    z_A = zbin_prop.z_A
    Deltanutilde = zbin_prop.Deltanutilde


    chi_A = zbin_prop.chi_A ; nutilde_A = zbin_prop.nutilde_A ; rnu_A = zbin_prop.rnu_A
    lambda_A = zbin_prop.lambda_A

    nu_tilde_min = zbin_prop.nu_tilde_min; nu_tilde_max = zbin_prop.nu_tilde_max
    z_A_min = zbin_prop.z_A_min ; z_A_max = zbin_prop.z_A_max

    chi_A_min = zbin_prop.chi_A_min; chi_A_max = zbin_prop.chi_A_max;
    delta_chipar_bin = zbin_prop.delta_chipar_bin

    FOV_A_sr = zbin_prop.FOV_A_sr
    delta_chiperp_bin = zbin_prop.delta_chiperp_bin
    apar = cosmological_parameters.apar ; aperp = cosmological_parameters.aperp


    qpar_min = zbin_prop.qpar_min ; qperp_min = zbin_prop.qperp_min
    num_qpar = 400 ; num_qperp = 399
    N_factor_integral_range=200
    qpar_arr = np.linspace(1e-7,N_factor_integral_range*qpar_min,num_qpar) ; qperp_arr = np.linspace(1e-7,N_factor_integral_range*qperp_min,num_qperp)


    K_21_qpar_arr = np.sin(qpar_arr*delta_chipar_bin/2.0)/ (qpar_arr*delta_chipar_bin/2.0)
    K_21_qperp_arr = np.sin(qperp_arr*delta_chiperp_bin/2.0)/(qperp_arr*delta_chiperp_bin/2.0)
    dK_21_qpar_arr = ( np.cos(qpar_arr*delta_chipar_bin/2.0)*qpar_arr*delta_chipar_bin/2.0 - np.sin(qpar_arr*delta_chipar_bin/2.0) )/(qpar_arr**2.*delta_chipar_bin/2.0)
    dK_21_qperp_arr = (np.cos(qperp_arr*delta_chiperp_bin/2.0)*qperp_arr*delta_chiperp_bin/2.0 - np.sin(qperp_arr*delta_chiperp_bin/2.0))/(qperp_arr**2.*delta_chiperp_bin/2.0)
    K_21_qpar_arr[0] = 1.0
    K_21_qperp_arr[0] = 1.0
    dK_21_qpar_arr[0] = 1.0
    dK_21_qperp_arr[0] = 1.0
    K_kappa_qperp_arr = K_21_qperp_arr
    K_kappa_qpar_arr = K_21_qpar_arr
    dK_kappa_qperp_arr = dK_21_qperp_arr
    dK_kappa_qpar_arr = dK_21_qpar_arr


    Int_alt_sinc_max = np.sin(np.amax(qperp_arr)*delta_chipar_bin/2.0)/ (np.amax(qperp_arr)*delta_chipar_bin/2.0)
    Int_alt_sinc_min = np.sin(qpar_min*delta_chipar_bin/2.0)/ (qpar_min*delta_chipar_bin/2.0)
    Int_alt_Pm_min = power_spec_functions.get_P_m_0(qpar_min)
    Int_alt_Pm_max = power_spec_functions.get_P_m_0(np.amax(qperp_arr))

    daperp_q_at_qmax = (aperp*np.amax(qperp_arr))**2. / (np.sqrt( np.amax(qperp_arr)**2. + np.amax(qpar_arr)**2.)  *aperp)
    dapar_q_at_qmax = (apar*np.amax(qpar_arr))**2. / (np.sqrt( np.amax(qperp_arr)**2. + np.amax(qpar_arr)**2.)  *apar)
    daperp_q_at_qmin = (aperp*qperp_arr[-1])**2. / (np.sqrt( qperp_arr[-1]**2. + qpar_arr[-1]**2.)  *aperp)
    dapar_q_at_qmin = (apar*qpar_arr[-1])**2. / (np.sqrt( qperp_arr[-1]**2. + qpar_arr[-1]**2.)  *apar)


    deriv_kernel_aperp = np.amax(qperp_arr)**2. * Int_alt_sinc_max**2. * Int_alt_Pm_max *daperp_q_at_qmax - qpar_min**2. *Int_alt_sinc_min**2. * Int_alt_Pm_min*daperp_q_at_qmin
    deriv_kernel_apar = np.amax(qperp_arr)**2. * Int_alt_sinc_max**2. * Int_alt_Pm_max *dapar_q_at_qmax - qpar_min**2. *Int_alt_sinc_min**2. * Int_alt_Pm_min*dapar_q_at_qmin

    K_21_qperp_2d_arr = np.outer(K_21_qperp_arr,np.ones(num_qpar))  ; K_21_qpar_2d_arr = np.outer(np.ones(num_qperp), K_21_qpar_arr)
    K_kappa_qperp_2d_arr = np.outer(K_kappa_qperp_arr,np.ones(num_qpar))  ; K_kappa_qpar_2d_arr = np.outer(np.ones(num_qperp), K_kappa_qpar_arr)

    dK_21_qperp_2d_arr = np.outer(dK_21_qperp_arr,np.ones(num_qpar))  ; dK_21_qpar_2d_arr = np.outer(np.ones(num_qperp), dK_21_qpar_arr)
    dK_kappa_qperp_2d_arr = np.outer(dK_kappa_qperp_arr,np.ones(num_qpar))  ; dK_kappa_qpar_2d_arr = np.outer(np.ones(num_qperp), dK_kappa_qpar_arr)


    q_par_2d_arr = np.outer(np.ones(num_qperp),qpar_arr)
    q_perp_2d_arr = np.outer(qperp_arr,np.ones(num_qpar))

    q_mag_2d_arr = np.sqrt(np.outer(qperp_arr,np.ones(num_qpar))**2 + np.outer(np.ones(num_qperp),qpar_arr)**2)
    fbao_q_2d_arr = power_spec_functions.get_fbao(q_mag_2d_arr)
    Pm_q_2d_arr = power_spec_functions.get_P_m_0(q_mag_2d_arr)

    Integral_ell_y_zA_over_qpar = np.trapz(0.5/np.pi**2 * Pm_q_2d_arr*K_21_qperp_2d_arr*K_kappa_qperp_2d_arr*K_21_qpar_2d_arr*K_kappa_qpar_2d_arr*q_perp_2d_arr, x=qperp_arr, axis=0)
    Integral_ell_y_zA_over_qpar_qperp = np.trapz(Integral_ell_y_zA_over_qpar, x=qpar_arr, axis=0)

    deriv_kernel_sln_aperp = (1/2.*np.pi)**2. *  deriv_kernel_aperp/Integral_ell_y_zA_over_qpar_qperp
    deriv_kernel_sln_apar = (1/2.*np.pi)**2. *  deriv_kernel_apar/Integral_ell_y_zA_over_qpar_qperp

    h=cosmological_parameters.h
    kP=np.loadtxt('Datafiles/P_k_nonlin_camb_z0.dat')  # goes up to k=18/Mpc
    P_spl_k=interp1d(kP[:,0]/h, kP[:,1]*h**3,bounds_error=False,fill_value=0.0)

    q_mag_arr_abs = qpar_arr
    dpk_dq = scipy.misc.derivative(power_spec_functions.get_P_m_0, q_mag_arr_abs,dx=1e-3)
    d_integrand = Pm_q_2d_arr*dK_21_qperp_2d_arr*dK_21_qpar_2d_arr*K_kappa_qperp_2d_arr*K_kappa_qpar_2d_arr\
                +Pm_q_2d_arr*K_21_qperp_2d_arr*K_21_qpar_2d_arr*dK_kappa_qperp_2d_arr*dK_kappa_qpar_2d_arr \
                + dpk_dq *K_21_qperp_2d_arr*K_21_qpar_2d_arr*K_kappa_qperp_2d_arr*K_kappa_qpar_2d_arr




    Integral_ell_y_zA_over_qpar_fbao = np.trapz(0.5/np.pi**2 * Pm_q_2d_arr * fbao_q_2d_arr/(1+fbao_q_2d_arr) *K_21_qperp_2d_arr*K_kappa_qperp_2d_arr*K_21_qpar_2d_arr*K_kappa_qpar_2d_arr*q_perp_2d_arr, x=qperp_arr, axis=0)
    Integral_ell_y_zA_over_qpar_qperp_fbao = np.trapz(Integral_ell_y_zA_over_qpar_fbao, x=qpar_arr, axis=0)


    Integral_ell_y_zA_coarse = Integral_ell_y_zA_over_qpar_qperp
    Integral_ell_y_zA_coarse_fbao = Integral_ell_y_zA_over_qpar_qperp_fbao

    C_ell_y_zA = np.zeros((ell_coarse_arr.size,y_coarse_arr.size))
    dPmdk_z0_ktot_2d_arr = np.zeros((ell_coarse_arr.size,y_coarse_arr.size))
    d2Pmdk_z0_ktot_2d_arr = np.zeros((ell_coarse_arr.size,y_coarse_arr.size))
    Pm_z0_ktot_2d_arr = np.zeros((ell_coarse_arr.size,y_coarse_arr.size))

    ell_coarse_min = np.min(ell_coarse_arr); ell_coarse_max = np.max(ell_coarse_arr)
    y_coarse_min = np.min(y_coarse_arr); y_coarse_max = np.max(y_coarse_arr)
    n_coarse_ell = ell_coarse_arr.size ; n_coarse_y = y_coarse_arr.size

   ### NOW COMPUTE 21-21-kappa BISPECTRUM

   # Set up 2d k arrays
    kperp_arr = ell_coarse_arr/chi_A; kpar_arr = y_coarse_arr/rnu_A
    ktot_2d_arr = np.sqrt(np.outer(ell_coarse_arr,np.ones(n_coarse_y))**2/chi_A**2 + np.outer(np.ones(n_coarse_ell),y_coarse_arr)**2/rnu_A**2)
    kpar_2d_arr = np.outer(np.ones(n_coarse_ell),kpar_arr)
    kperp_2d_arr = np.outer(kperp_arr,np.ones(n_coarse_y))
    mu_k_2d_arr = kpar_2d_arr/ktot_2d_arr
    fbao_2d_k_arr = power_spec_functions.get_fbao(ktot_2d_arr)

   # GET P_m DERIVATIVE
    k_arr = kpar_arr   # USE kpar_arr to get derivative since y_arr gives smaller kmin and larger kmax
    Pm_z0_karr=power_spec_functions.get_P_m_0(k_arr)
    dPmdk_z0_arr=scipy.misc.derivative(power_spec_functions.get_P_m_0, k_arr,dx=1e-3)
    dPmdk_z0_arr_interp = scipy.interpolate.interp1d(dPmdk_z0_arr,k_arr,bounds_error=False,fill_value=0.0)
    d2Pmdk_z0_arr=scipy.misc.derivative(dPmdk_z0_arr_interp, k_arr,dx=1e-3)
    dfbao_dk = scipy.misc.derivative(power_spec_functions.get_fbao, k_arr,dx=1e-3)

   # Now evaluate all terms in bispectrum

    Pm_z0_ktot_2d_arr[:,] = np.interp(ktot_2d_arr[:,],k_arr,Pm_z0_karr)
    dPmdk_z0_ktot_2d_arr[:,]= np.interp(ktot_2d_arr[:,],k_arr,dPmdk_z0_arr)
    d2Pmdk_z0_ktot_2d_arr[:,] =np.interp(ktot_2d_arr[:,],k_arr,d2Pmdk_z0_arr)

    F_bias_rsd_2d_arr = (power_spec_functions.get_HI_bias(z_A, bias_var)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)

    ##***********************************************##
    ## *** Updated Bispec kernel *****
    ##***********************************************##

    b2 = power_spec_functions.get_HI_bias_2nd_order(z_A)
    f = power_spec_functions.get_growth_factor_f(z_A, gamma_var)
    mk = mu_k_2d_arr
    b1 = power_spec_functions.get_HI_bias(z_A, bias_var)
    Z1 = (b1 + f*mk**2.)

    Updated_derivative_term = (1/3.)*(f*mk**2. - mk**2. + 2.)*(3.0 - dPmdk_z0_ktot_2d_arr*ktot_2d_arr/Pm_z0_ktot_2d_arr)
    Updated_derivative_independent_term = (1/(14*Z1))*(14*b1*f*mk**2. + 14*b1*(f/3.) + 26.*b1*(mk**2./3.) + 26.*(b1/3.) + 28*b2 + 14*f**2.*mk**4. - 14*f**2*mk**2  \
    -6*f*mk**4. + 38*f*(mk**2/3.))

    Kernel_2d_arr_updated = Updated_derivative_term + Updated_derivative_independent_term


    Kernel_2d_arr = Kernel_2d_arr_updated


    P_21_z0_2d_arr = F_bias_rsd_2d_arr**2 *power_spec_functions.get_P_m_0(ktot_2d_arr) * power_spec_functions.get_mean_temp(z_A)**2

    volume_A_max = cd.comoving_volume(z_A_max, **cosmo); volume_A_min = cd.comoving_volume(z_A_min, **cosmo)
    volume_A = (volume_A_max - volume_A_min) *FOV_A_sr/(4.0*np.pi)


    W_kappa_over_chi_sq=lensing_kernel.get_lensing_kernel_real_sp(chi_A) / chi_A**2

    Volume_lensing_kernel_growth_factor = W_kappa_over_chi_sq * power_spec_functions.get_growth_function_D(z_A)**4 * volume_A / (chi_A**2 * rnu_A)

    C_ell_y_zA =  Integral_ell_y_zA_coarse* P_21_z0_2d_arr * Volume_lensing_kernel_growth_factor * Kernel_2d_arr
    C_ell_y_zA_fbao =  Integral_ell_y_zA_coarse_fbao* P_21_z0_2d_arr * Volume_lensing_kernel_growth_factor * Kernel_2d_arr

    f_zA = power_spec_functions.get_growth_factor_f(z_A, bias_var)
    drsd_du2 = 2.*f_zA/ F_bias_rsd_2d_arr

    dlogpk_dk = Fpm.logpk_derivative(P_spl_k, ktot_2d_arr) # Numerical deriv.
    daperp_u2 = -2. * (kperp_2d_arr/kpar_2d_arr * aperp/apar * mu_k_2d_arr**2.)**2. / aperp
    dapar_u2 =   2. * (kperp_2d_arr/kpar_2d_arr* aperp/apar * mu_k_2d_arr**2.)**2. / apar
    daperp_k = (aperp*kperp_2d_arr)**2. / (ktot_2d_arr*aperp)
    daperp_q = (aperp*q_perp_2d_arr)**2. / (q_mag_2d_arr*aperp)
    dapar_k  = (apar*kpar_2d_arr)**2. / (ktot_2d_arr*apar)
    dapar_q = (apar*q_par_2d_arr)**2. / (q_mag_2d_arr*apar)
    dKernel_term_1=4.*power_spec_functions.get_growth_factor_f(z_A, gamma_var)*(1+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)
    dKernel_term_2 = 2*power_spec_functions.get_growth_factor_f(z_A, gamma_var)**2. * (2*mu_k_2d_arr**2-1.0)
    dKernel_term_3 = power_spec_functions.get_growth_factor_f(z_A, gamma_var)*(power_spec_functions.get_HI_bias_2nd_order(z_A)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*(2*mu_k_2d_arr**2-1.0)*2*(1+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2))
    dKernel_term_4 = 0.5*power_spec_functions.get_growth_factor_f(z_A, gamma_var)*(3.0 - dPmdk_z0_ktot_2d_arr*ktot_2d_arr/Pm_z0_ktot_2d_arr)
    dkernel_2d_arr = (((dKernel_term_1 + dKernel_term_2)*F_bias_rsd_2d_arr - dKernel_term_3)/F_bias_rsd_2d_arr**2.) + dKernel_term_4
    dlogPm_dlogk = d2Pmdk_z0_ktot_2d_arr * (ktot_2d_arr/Pm_z0_ktot_2d_arr) + dPmdk_z0_ktot_2d_arr*(Pm_z0_ktot_2d_arr-ktot_2d_arr*dPmdk_z0_ktot_2d_arr)/Pm_z0_ktot_2d_arr**2.
    ### For nonlinear cases sigma_nl must add extra terms dlogpk_dk + drsd_dk + dbias_k below as in Bull et al.
    dIntegral_ell_y_zA_over_qpar_aperp = np.trapz(0.5/np.pi**2 * d_integrand *q_perp_2d_arr*daperp_q, x=qperp_arr, axis=0)
    dIntegral_ell_y_zA_over_qpar_qperp_aperp = np.trapz(dIntegral_ell_y_zA_over_qpar_aperp, x=qpar_arr, axis=0)
    dIntegral_ell_y_zA_alternative = 2.*np.pi**2. * np.amax(Pm_q_2d_arr*K_21_qperp_2d_arr*K_kappa_qperp_2d_arr*K_21_qpar_2d_arr*K_kappa_qpar_2d_arr*q_perp_2d_arr*dapar_q)

    dIntegral_ell_y_zA_over_qpar_apar= np.trapz(0.5/np.pi**2 * d_integrand *q_perp_2d_arr*dapar_q, x=qperp_arr, axis=0)
    dIntegral_ell_y_zA_over_qpar_qperp_apar = np.trapz(dIntegral_ell_y_zA_over_qpar_apar, x=qpar_arr, axis=0)
    d2Pm_dk_aperp_apar = d2Pmdk_z0_ktot_2d_arr*(ktot_2d_arr/Pm_z0_ktot_2d_arr) + dPmdk_z0_ktot_2d_arr*(Pm_z0_ktot_2d_arr-ktot_2d_arr*dPmdk_z0_ktot_2d_arr)/(Pm_z0_ktot_2d_arr**2.)

    ##***********************************************##
    ## *** Updated Derivatives for kernel *****
    ##***********************************************##

    d_derivative_term_dmu = (3.0 - dPmdk_z0_ktot_2d_arr*ktot_2d_arr/Pm_z0_ktot_2d_arr)*(2.*f*mk/3. - 2.*mk/3.)
    d_derivative_independent_term_dmu =  (1./(Z1*14.))*(-b1*(-14.*f*mk - 26.*mk/3.) + b1*(14.*f*mk + 26.*mk/3.) - f*mk*(-21.*f*mk**2. + 7.*f + 9.*mk**2. - 19./3.) + f*mk*(21.*f*mk**2. - 7.*f - 9.*mk**2. + 19./3.) - f*(-7.*f*mk**3. + 7.*f*mk + 3.*mk**3. - 19.*mk/3.) + f*(7.*f*mk**3. - 7.*f*mk - 3.*mk**3. + 19.*mk/3.))
    dkernel_2d_arr_dmu = 2*mk*(d_derivative_term_dmu  + d_derivative_independent_term_dmu )


    deriv_P21_aperp_over_P21 = ( (2./aperp) + drsd_du2 * daperp_u2 \
                        + (dlogpk_dk)*daperp_k )
    deriv_P21_apar_over_P21 =  ( (1./apar)  + drsd_du2 * dapar_u2 \
                        + (dlogpk_dk)*dapar_k  )

    deriv_aperp = ( (4./aperp) + deriv_P21_aperp_over_P21  + deriv_kernel_sln_aperp \
        +(dkernel_2d_arr_dmu*daperp_u2-dlogPm_dlogk*0.5*(1+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)*daperp_k )\
			/Kernel_2d_arr ) * C_ell_y_zA #+ dIntegral_ell_y_zA_over_qpar_qperp_aperp/Integral_ell_y_zA_coarse
    deriv_apar =   ( (1./apar) + deriv_P21_apar_over_P21  + deriv_kernel_sln_apar \
        +(dkernel_2d_arr_dmu*dapar_u2-dlogPm_dlogk *0.5*(1+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)*dapar_k )\
			/Kernel_2d_arr ) * C_ell_y_zA #+ dIntegral_ell_y_zA_over_qpar_qperp_apar/Integral_ell_y_zA_coarse


    Abao_zA,sig8, b1_zA, b2_zA, f_zA, aperp, apar= fid_params
    sig8_zA = sig8*power_spec_functions.get_growth_function_D(z_A)


    dB_ell_y_dAbao = C_ell_y_zA*(fbao_2d_k_arr/(1.+fbao_2d_k_arr) + Integral_ell_y_zA_coarse_fbao/Integral_ell_y_zA_coarse \
							-0.5*(1+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)\
							*(ktot_2d_arr/ (1+power_spec_functions.get_growth_factor_f(z_A, gamma_var))**2. )*dfbao_dk  )

    sig8_deriv_kernel = 4.0 * Kernel_2d_arr / (sig8_zA*power_spec_functions.get_growth_function_D(z_A))
    dB_ell_y_dsig8_zA = C_ell_y_zA * sig8_deriv_kernel/Kernel_2d_arr

    dB_ell_y_db2_zA = (2/Z1)* C_ell_y_zA/Kernel_2d_arr

    dB_ell_y_db1_zA = ((14.*f*mk**2. + 14.*f/3. + 26.*mk**2./3. + 26./3.)/14./(14.*Z1))* C_ell_y_zA/Kernel_2d_arr

    f_deriv_kernel =(1/(14.*Z1))*(-b1*(-7.*mk**2. - 7./3.) + b1*(7.*mk**2. + 7./3.) - f*mk*(-7.*mk**3. + 7.*mk) + f*mk*(7.*mk**3. - 7.*mk) - mk*(-7.*f*mk**3. + 7.*f*mk + 3.*mk**3. - 19.*mk/3.) + mk*(7.*f*mk**3. - 7.*f*mk - 3.*mk**3. + 19.*mk/3.))
    dB_ell_y_df_zA =  C_ell_y_zA*(2*mu_k_2d_arr**2/F_bias_rsd_2d_arr + f_deriv_kernel/Kernel_2d_arr)

    dC_dns = np.log(ktot_2d_arr/0.05)*C_ell_y_zA


    return dB_ell_y_dAbao,dB_ell_y_dsig8_zA, dB_ell_y_db1_zA ,dC_dns, dB_ell_y_df_zA, deriv_aperp, deriv_apar


def get_three_param_FisherMatrix_Cl_21_21_auto_LCDM(zbin_prop, num_k_bins_ell = 15, num_k_bins_y=14):

	## SET Z BIN PROPERTIES

    Deltanutilde = zbin_prop.Deltanutilde    # Bin width
    deltanutilde = zbin_prop.deltanutilde	 # Channel width

    z_A = zbin_prop.z_A

    nutilde_A = zbin_prop.nutilde_A
    chi_A=zbin_prop.chi_A
    rnu_A=zbin_prop.rnu_A
    lambda_A = zbin_prop.lambda_A
    FOV_A_sr = zbin_prop.FOV_A_sr
    theta_min_sr = zbin_prop.theta_min_sr

	### SET RADIAL MODES

    y_min = zbin_prop.y_min_expt ; y_max = zbin_prop.y_max_expt
    result = zbin_prop.get_y_arr_expt() ; n_y = result[0]  ; y_arr = result[1]

	### SET TRANSVERSE MODES

    ell_min = zbin_prop.ell_min_expt  ; ell_max = zbin_prop.ell_max_expt
    result = zbin_prop.get_ell_arr_expt() ; n_ell = result[0]  ; ell_arr = result[1]   # CHANGE THESE TO _expt

    N_patches = zbin_prop.N_patches
    Mode_Volume_Factor = zbin_prop.Mode_Volume_Factor

	### COMPUTE AUTO-21-21 Fisher matrix for Tb, f, b1,b2 - MERGE ALL FISHER MATRIX CALCS INTO ONE METHOD


    aperp = cosmological_parameters.aperp
    apar = cosmological_parameters.apar
    sig8 = cosmological_parameters.sig8
    Abao = cosmological_parameters.Abao
    f_nl = cosmological_parameters.f_nl

    ell_min_ltd = ell_min ; y_min_ltd = y_min

    non_linear_scale = zbin_prop.knl
    ell_max_ltd = non_linear_scale*chi_A
    y_max_ltd = non_linear_scale*rnu_A

    ell_arr_ltd = np.linspace(ell_min_ltd,ell_max_ltd,n_ell); y_arr_ltd = np.linspace(y_min_ltd,y_max_ltd,n_y)

    f_zA = power_spec_functions.get_growth_factor_f(z_A, gamma_var=1.0)
    b1_zA=power_spec_functions.get_HI_bias(z_A, bias_var=1.0); b2_zA = power_spec_functions.get_HI_bias_2nd_order(z_A)


    fid_fisher_params = np.array([Abao,sig8, b1_zA, b2_zA,f_zA, aperp, apar])
    n_fisher_params = fid_fisher_params.size

    Cl_Abao,Cl_sig8, Cl_b1, Cl_b2, Cl_f, Cl_aperp, Cl_apar = Cl21_Distance_derivs_LCDM(zbin_prop, ell_arr_ltd, y_arr_ltd, fid_fisher_params, bias_var=1., gamma_var=1.)

    Cl_signal = Fpm.Cl_21_auto_ell_y_zA(ell_arr_ltd, y_arr_ltd, zbin_prop, bias_var=1., gamma_var=1.)

    parameter_deriv_C_ell_y_arr = np.array([Cl_Abao,Cl_sig8, Cl_b1, Cl_b2, Cl_f, Cl_aperp, Cl_apar])


    ell_2d_arr_ltd=np.outer(ell_arr_ltd,np.ones(n_y))

    Cl_21_noise_ell_arr=Fpm.HI_angular_noise_ell_y(ell_arr_ltd, zbin_prop, experiment_name='hirax', mode='interferometer', show_plots=False)
    Cl_21_noise_ell_y_2d_arr=  np.outer(Cl_21_noise_ell_arr,np.ones(n_y))
    Cl_21_noise_ell_y_2d_arr_allsky = Fpm.HI_angular_noise_ell_y_allsky(Cl_21_noise_ell_y_2d_arr)

    Cl_21_auto_ell_y_2d_arr=Fpm.Cl_21_auto_ell_y_zA(ell_arr_ltd, y_arr_ltd, zbin_prop, bias_var=1., gamma_var=1.) # 21cm angular PS at z vs ell,y

    variance_arr = (Cl_21_noise_ell_y_2d_arr_allsky + Cl_21_auto_ell_y_2d_arr)**2


    fisher_arr_zA_21_21 = np.zeros((n_fisher_params,n_fisher_params)) ; corr_arr_zA_21_21 = np.zeros((n_fisher_params,n_fisher_params))

    for ii, pp_i in enumerate(fid_fisher_params):
		deriv_arr_ii = parameter_deriv_C_ell_y_arr[ii,]
		for jj, pp_j in enumerate(fid_fisher_params):
			deriv_arr_jj = parameter_deriv_C_ell_y_arr[jj,]
			signal_arr_sq = deriv_arr_ii * deriv_arr_jj
			fisher_ratio_arr = signal_arr_sq/variance_arr
			fisher_arr_zA_21_21[ii,jj] = 0.5 * Mode_Volume_Factor/(2*np.pi**2) * np.trapz(np.trapz(fisher_ratio_arr*ell_2d_arr_ltd, ell_arr_ltd, axis=0), y_arr_ltd, axis=0)


    fid_param_label = ['Abao','sig8','b1','b2','f', 'aperp', 'apar']

    return fid_param_label, fid_fisher_params, fisher_arr_zA_21_21



def get_Clkappa_distance_params_fisher_LCDM(zbin_prop, num_k_bins_ell = 15, num_k_bins_y=14):

	## SET Z BIN PROPERTIES

    Deltanutilde = zbin_prop.Deltanutilde    # Bin width
    deltanutilde = zbin_prop.deltanutilde	 # Channel width

    z_A = zbin_prop.z_A

    nutilde_A = zbin_prop.nutilde_A
    chi_A=zbin_prop.chi_A
    rnu_A=zbin_prop.rnu_A
    lambda_A = zbin_prop.lambda_A
    FOV_A_sr = zbin_prop.FOV_A_sr
    theta_min_sr = zbin_prop.theta_min_sr

    aperp = cosmological_parameters.aperp
    apar = cosmological_parameters.apar
    sig8 = cosmological_parameters.sig8
    Abao = cosmological_parameters.Abao


	### SET RADIAL MODES

    y_min = zbin_prop.y_min_expt ; y_max = zbin_prop.y_max_expt
    result = zbin_prop.get_y_arr_expt() ; n_y = result[0]  ; y_arr = result[1]

	### SET TRANSVERSE MODES

    ell_min = zbin_prop.ell_min_expt  ; ell_max = zbin_prop.ell_max_expt
    result = zbin_prop.get_ell_arr_expt() ; n_ell = result[0]  ; ell_arr = result[1]   # CHANGE THESE TO _expt

    N_patches = zbin_prop.N_patches
    Mode_Volume_Factor = zbin_prop.Mode_Volume_Factor

    ### COMPUTE AUTO-21-21 Fisher matrix for Tb, f, b1,b2 - MERGE ALL FISHER MATRIX CALCS INTO ONE METHOD

    sig8=cosmological_parameters.sig8
    h = cosmological_parameters.h
    Om_m = cosmological_parameters.om_m
    ns = cosmological_parameters.ns

    ell_min_ltd = ell_min ; y_min_ltd = y_min
    non_linear_scale = zbin_prop.knl
    ell_max_ltd = non_linear_scale*chi_A
    y_max_ltd = non_linear_scale*rnu_A
    ell_arr_ltd = np.linspace(ell_min_ltd,ell_max_ltd,n_ell); y_arr_ltd = np.linspace(y_min_ltd,y_max_ltd,n_y)

    Omega_HI_zA=power_spec_functions.get_Omega_HI(z_A) ; f_zA = power_spec_functions.get_growth_factor_f(z_A, gamma_var=1.0)
    b1_zA=power_spec_functions.get_HI_bias(z_A, bias_var=1.0); b2_zA = power_spec_functions.get_HI_bias_2nd_order(z_A)

    fid_fisher_params = np.array([sig8, ns, Om_m, h])
    n_fisher_params = fid_fisher_params.size

    Cl_sig8, Cl_ns, Cl_Om, Cl_h = Cl_kappa_distance_derivs_LCDM(ell_arr_ltd, y_arr_ltd, zbin_prop, fid_fisher_params, bias_var=1., gamma_var=1.)

    Cl_signal = Fpm.CMB_convergence_power(ell_arr_ltd)

    parameter_deriv_C_ell_y_arr = np.array([Cl_sig8, Cl_ns, Cl_Om, Cl_h])

    ell_2d_arr_ltd=np.outer(ell_arr_ltd,np.ones(n_y))

    Cl_kappa_noise_ell_arr=Fpm.CMB_convergence_noise(ell_arr_ltd)
    Cl_kappa_noise_ell_y_2d_arr=  np.outer(Cl_kappa_noise_ell_arr,np.ones(n_y))
    Cl_kappa_auto_ell_y_2d_arr= np.outer(Cl_signal,np.ones(n_y))

    variance_arr = (Cl_kappa_noise_ell_y_2d_arr+ Cl_kappa_auto_ell_y_2d_arr)**2.

    fisher_arr_zA_kappa = np.zeros((n_fisher_params,n_fisher_params)) ; corr_arr_zA_kappa = np.zeros((n_fisher_params,n_fisher_params))

    for ii, pp_i in enumerate(fid_fisher_params):
		deriv_arr_ii = parameter_deriv_C_ell_y_arr[ii,]
		for jj, pp_j in enumerate(fid_fisher_params):
			deriv_arr_jj = parameter_deriv_C_ell_y_arr[jj,]
			signal_arr_sq = deriv_arr_ii * deriv_arr_jj
			fisher_ratio_arr = signal_arr_sq/variance_arr
			fisher_arr_zA_kappa[ii,jj] = 0.5 * Mode_Volume_Factor/(2.*np.pi**2) * np.trapz(np.trapz(fisher_ratio_arr*ell_2d_arr_ltd, ell_arr_ltd, axis=0), y_arr_ltd, axis=0)

    fid_param_label = ['sig8', 'ns', 'Om_m', 'h']

    return fid_param_label, fid_fisher_params, fisher_arr_zA_kappa


def get_distance_params_bispec_fisher_LCDM(zbin_prop, num_k_bins_ell = 15, num_k_bins_y=14):

	## SET Z BIN PROPERTIES

    Deltanutilde = zbin_prop.Deltanutilde    # Bin width
    deltanutilde = zbin_prop.deltanutilde	 # Channel width

    z_A = zbin_prop.z_A

    nutilde_A = zbin_prop.nutilde_A
    chi_A=zbin_prop.chi_A
    rnu_A=zbin_prop.rnu_A
    lambda_A = zbin_prop.lambda_A
    FOV_A_sr = zbin_prop.FOV_A_sr
    theta_min_sr = zbin_prop.theta_min_sr

    aperp = cosmological_parameters.aperp
    apar = cosmological_parameters.apar
    sig8 = cosmological_parameters.sig8
    Abao = cosmological_parameters.Abao
    fnl = cosmological_parameters.f_nl


	### SET RADIAL MODES

    y_min = zbin_prop.y_min_expt ; y_max = zbin_prop.y_max_expt
    result = zbin_prop.get_y_arr_expt() ; n_y = result[0]  ; y_arr = result[1]

	### SET TRANSVERSE MODES

    ell_min = zbin_prop.ell_min_expt  ; ell_max = zbin_prop.ell_max_expt
    result = zbin_prop.get_ell_arr_expt() ; n_ell = result[0]  ; ell_arr = result[1]   # CHANGE THESE TO _expt

    N_patches = zbin_prop.N_patches
    Mode_Volume_Factor = zbin_prop.Mode_Volume_Factor

	### GET 21-21 Auto Power Spectrum - signal and noise
    Cl_21_auto_ell_y_2d_arr=Fpm.Cl_21_auto_ell_y_zA(ell_arr, y_arr, zbin_prop, bias_var=1., gamma_var=1.) # 21cm angular PS at z vs ell,y
    Cl_21_noise_ell_arr=Fpm.HI_angular_noise_ell_y(ell_arr, zbin_prop, experiment_name='hirax', mode='interferometer', show_plots=False)
    Cl_21_noise_ell_y_2d_arr=  np.outer(Cl_21_noise_ell_arr,np.ones(n_y))

    CMB_kappa_average_variance_sky = Fpm.compute_average_kappa_variance_sky_zA(zbin_prop, bias_var=1., gamma_var=1.)


    ell_min_ltd = ell_min ; y_min_ltd = y_min
    non_linear_scale = zbin_prop.knl
    ell_max_ltd = non_linear_scale*chi_A
    y_max_ltd = non_linear_scale*rnu_A
    ell_arr_ltd = np.linspace(ell_min_ltd,ell_max_ltd,n_ell); y_arr_ltd = np.linspace(y_min_ltd,y_max_ltd,n_y)

    Omega_HI_zA=power_spec_functions.get_Omega_HI(z_A) ; f_zA = power_spec_functions.get_growth_factor_f(z_A, gamma_var=1.0)
    b1_zA=power_spec_functions.get_HI_bias(z_A, bias_var=1.0); b2_zA = power_spec_functions.get_HI_bias_2nd_order(z_A)

    fid_fisher_params = np.array([Abao,sig8, b1_zA, b2_zA, f_zA, aperp, apar])
    n_fisher_params = fid_fisher_params.size

    print "STARTING FISHER MATRIX"

    Bl_Abao,Bl_sig8,Bl_b1,Bl_b2 ,Bl_f,Bl_aperp, Bl_apar =  bispec_distance_derivs_LCDM(ell_arr_ltd, y_arr_ltd, zbin_prop, fid_fisher_params, bias_var=1., gamma_var=1.)

    Bl_signal = Fpm.integ_bispec_kappa_21_21_ell_y_zA(ell_arr_ltd, y_arr_ltd, zbin_prop, bias_var=1., gamma_var=1.)


    parameter_deriv_ell_y_arr = np.array([Bl_Abao,Bl_sig8,Bl_b1,Bl_b2,Bl_f,Bl_aperp, Bl_apar])#,Bl_Omega_HI # parameter_deriv_ell_y_arr[ii,]

    ell_2d_arr_ltd=np.outer(ell_arr_ltd,np.ones(n_y))
    CMB_kappa_noise_ell = Fpm.CMB_convergence_noise(ell_2d_arr_ltd)
    CMB_kappa_signal_ell = Fpm.CMB_convergence_power(ell_2d_arr_ltd)

    Cl_21_noise_ell_arr=Fpm.HI_angular_noise_ell_y(ell_arr_ltd, zbin_prop, experiment_name='hirax', mode='interferometer', show_plots=False)
    Cl_21_noise_ell_y_2d_arr=  np.outer(Cl_21_noise_ell_arr,np.ones(n_y))
    Cl_21_noise_ell_y_2d_arr_allsky = Cl_21_noise_ell_y_2d_arr/fsky

    variance_arr = 6.* N_patches**2. * Bl_signal**2. + \
    (3.*fsky**2.)*(Cl_21_noise_ell_y_2d_arr_allsky + Cl_21_auto_ell_y_2d_arr)**2. * CMB_kappa_average_variance_sky # (CMB_kappa_noise_ell+CMB_kappa_signal_ell)
    fisher_arr_zA_21_21_kappa = np.zeros((n_fisher_params,n_fisher_params)) ; corr_arr_zA=np.zeros((n_fisher_params,n_fisher_params))
    variance_arr = variance_arr


    for ii, pp_i in enumerate(fid_fisher_params):
		deriv_arr_ii = parameter_deriv_ell_y_arr[ii,]
		for jj, pp_j in enumerate(fid_fisher_params):
			deriv_arr_jj = parameter_deriv_ell_y_arr[jj,]
			signal_arr_sq = deriv_arr_ii * deriv_arr_jj
			fisher_ratio_arr = N_patches**2 * signal_arr_sq/variance_arr
			fisher_arr_zA_21_21_kappa[ii,jj] = 0.5 * Mode_Volume_Factor/(2*np.pi**2) * np.trapz(np.trapz(fisher_ratio_arr*ell_2d_arr_ltd, ell_arr_ltd, axis=0), y_arr_ltd, axis=0)


    fid_param_label = ['Abao','sig8', 'b1', 'b2', 'f', 'aperp', 'apar']


    return fid_param_label, fid_fisher_params, fisher_arr_zA_21_21_kappa




################################################################################################
# Below are all w0waCDM derivatives and Fisher Matrices
################################################################################################


def Cl21_Distance_derivs_w0waCDM(zbin_prop,ell_arr, y_arr,fid_params, bias_var=1., gamma_var=1.):

   n_ell = ell_arr.size ; n_y = y_arr.size
   z_A = zbin_prop.z_A
   chi_A = zbin_prop.chi_A ; nutilde_A = zbin_prop.nutilde_A ; rnu_A = zbin_prop.rnu_A
   kperp_arr = ell_arr/chi_A; kpar_arr = y_arr/rnu_A
   apar = cosmological_parameters.apar ; aperp = cosmological_parameters.aperp

   ktot_2d_arr = np.sqrt(np.outer(ell_arr,np.ones(n_y))**2/chi_A**2 + np.outer(np.ones(n_ell),y_arr)**2/rnu_A**2)
   kpar_2d_arr = np.outer(np.ones(n_ell),kpar_arr)
   kperp_2d_arr = np.outer(kperp_arr,np.ones(n_y))
   mu_k_2d_arr = kpar_2d_arr/ktot_2d_arr
   fbao_2d_arr = power_spec_functions.get_fbao(ktot_2d_arr)
   alpha_fnl_2d_arr = 3e5**2.*2.*ktot_2d_arr**2.* power_spec_functions.get_transfer_function(ktot_2d_arr)*power_spec_functions.get_growth_function_D(z_A)\
    				  /(3. * cosmological_parameters.om_m * (cosmological_parameters.H0)**2.)
   beta_fnl = 2.*cosmological_parameters.delta_c_fnl* ( power_spec_functions.get_HI_bias(z_A,bias_var) - 1.)


   F_bias_rsd_2d_arr = power_spec_functions.get_HI_bias(z_A, bias_var)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2
   F_bias_rsd_sq_2d_arr = F_bias_rsd_2d_arr**2
   F_bias_rsd_fnl_2d_arr = power_spec_functions.get_HI_bias(z_A, bias_var)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2 + \
   cosmological_parameters.f_nl * beta_fnl / alpha_fnl_2d_arr

   P_21_tot_2d_arr = F_bias_rsd_sq_2d_arr *power_spec_functions.get_P_m_0(ktot_2d_arr) * \
	power_spec_functions.get_mean_temp(z_A)**2 * power_spec_functions.get_growth_function_D(z_A)**2  # mK^2
   P_21_tot_2d_arr_fnl =  F_bias_rsd_fnl_2d_arr*power_spec_functions.get_P_m_0(ktot_2d_arr) * \
 	power_spec_functions.get_mean_temp(z_A)**2 * power_spec_functions.get_growth_function_D(z_A)**2

   Cl_21_auto_ell_y_2d_arr = P_21_tot_2d_arr / (chi_A**2 * rnu_A)
   Cl_21_auto_ell_y_2d_arr_fnl = P_21_tot_2d_arr_fnl / (chi_A**2 * rnu_A)

   f_zA = power_spec_functions.get_growth_factor_f(z_A, bias_var)
   drsd_du2 = 2.*f_zA/ F_bias_rsd_2d_arr

   h=cosmological_parameters.h
   kP=np.loadtxt('Datafiles/P_k_nonlin_camb_z0.dat')  # goes up to k=18/Mpc
   P_spl_k=interp1d(kP[:,0]/h, kP[:,1]*h**3,bounds_error=False,fill_value=0.0)


   dlogpk_dk = Fpm.logpk_derivative(P_spl_k, ktot_2d_arr) # Numerical deriv.
   daperp_u2 = -2. * (kperp_2d_arr/kpar_2d_arr * aperp/apar * mu_k_2d_arr**2.)**2. / aperp
   dapar_u2 =   2. * (kperp_2d_arr/kpar_2d_arr* aperp/apar * mu_k_2d_arr**2.)**2. / apar
   daperp_k = (aperp*kperp_2d_arr)**2. / (ktot_2d_arr*aperp)
   dapar_k  = (apar*kpar_2d_arr)**2. / (ktot_2d_arr*apar)

   ### For nonlinear cases sigma_nl must add extra terms dlogpk_dk + drsd_dk + dbias_k below as in Bull et al.
   deriv_aperp = ( (2./aperp) + drsd_du2 * daperp_u2 \
                       + (dlogpk_dk)*daperp_k ) * Cl_21_auto_ell_y_2d_arr
   deriv_apar =  ( (1./apar)  + drsd_du2 * dapar_u2 \
                       + (dlogpk_dk)*dapar_k  ) * Cl_21_auto_ell_y_2d_arr

   deriv_f = 2.*mu_k_2d_arr**2./ F_bias_rsd_2d_arr *  Cl_21_auto_ell_y_2d_arr

   Abao_zA,sig8, b1_zA, b2_zA,  f_zA, aperp, apar= fid_params
   sig8_zA = sig8*power_spec_functions.get_growth_function_D(z_A)

   dC_ell_y_dAbao = Cl_21_auto_ell_y_2d_arr*fbao_2d_arr/(1.+fbao_2d_arr)

   sig8_deriv_kernel = 2.0/(sig8_zA*power_spec_functions.get_growth_function_D(z_A))
   dC_ell_y_dsig8_zA = Cl_21_auto_ell_y_2d_arr * sig8_deriv_kernel

   b2_deriv_kernel = 0.0
   dC_ell_y_db2_zA = Cl_21_auto_ell_y_2d_arr * b2_deriv_kernel

   b1_deriv_kernel = 2.0/F_bias_rsd_2d_arr
   dC_ell_y_db1_zA = Cl_21_auto_ell_y_2d_arr*b1_deriv_kernel

   fnl_deriv_kernel = 2.0*beta_fnl/(alpha_fnl_2d_arr*F_bias_rsd_fnl_2d_arr)
   dC_ell_y_dfnl_zA = Cl_21_auto_ell_y_2d_arr*fnl_deriv_kernel

   return  dC_ell_y_dAbao, dC_ell_y_dsig8_zA, dC_ell_y_db1_zA, dC_ell_y_db2_zA, deriv_f, deriv_aperp, deriv_apar



def Cl_kappa_distance_derivs_w0waCDM(ell_arr, y_arr, zbin_prop, fid_params, bias_var=1., gamma_var=1.): # 21cm angular PS at z vs ell,y

    n_ell = ell_arr.size ; n_y = y_arr.size
    z_A = zbin_prop.z_A
    chi_A = zbin_prop.chi_A ; nutilde_A = zbin_prop.nutilde_A ; rnu_A = zbin_prop.rnu_A
    chi_max = zbin_prop.chi_A_min; chi_min = zbin_prop.chi_A_max
    kperp_arr = ell_arr/chi_A; kpar_arr = y_arr/rnu_A
    apar = cosmological_parameters.apar ; aperp = cosmological_parameters.aperp


    ktot_2d_arr = np.sqrt(np.outer(ell_arr,np.ones(n_y))**2/chi_A**2 + np.outer(np.ones(n_ell),y_arr)**2/rnu_A**2)
    kpar_2d_arr = np.outer(np.ones(n_ell),kpar_arr)
    kperp_2d_arr = np.outer(kperp_arr,np.ones(n_y))
    fbao_2d_arr = power_spec_functions.get_fbao(ktot_2d_arr)


    Cl_kappa_auto_ell_y_2d_arr = Fpm.CMB_convergence_power(kperp_2d_arr*chi_A)


	# COMPUTE Cl_21 DERIVATIVES
    f_zA = power_spec_functions.get_growth_factor_f(z_A, bias_var)

    h=cosmological_parameters.h
    kP=np.loadtxt('Datafiles/P_k_nonlin_camb_z0.dat')  # goes up to k=18/Mpc
    P_spl_k=interp1d(kP[:,0]/h, kP[:,1]*h**3,bounds_error=False,fill_value=0.0)


    dlogpk_dk = Fpm.logpk_derivative(P_spl_k, ktot_2d_arr) # Numerical deriv.
    daperp_k = (aperp*kperp_2d_arr)**2. / (ktot_2d_arr*aperp)
    dapar_k  = (apar*kpar_2d_arr)**2. / (ktot_2d_arr*apar)

   ### For nonlinear cases sigma_nl must add extra terms dlogpk_dk + drsd_dk + dbias_k below as in Bull et al.
    W_kappa_over_chi_sq=lensing_kernel.get_lensing_kernel_real_sp(chi_A)**2.*power_spec_functions.get_growth_function_D(z_A)**2. / chi_A**2.


    sig8, Om_DE, h, w0, wa = fid_params
    sig8_zA = sig8
    Om_m = 1 - Om_DE


    sig8_deriv_kernel = 2.0/(sig8_zA)
    dC_ell_y_dsig8_zA = Cl_kappa_auto_ell_y_2d_arr * sig8_deriv_kernel

    dC_ell_y_dOm_m = 2.0*Cl_kappa_auto_ell_y_2d_arr/(Om_m)
    dC_ell_y_dOm_DE = -2.0*Cl_kappa_auto_ell_y_2d_arr/(Om_m)


    dC_ell_y_dh = 4.0*Cl_kappa_auto_ell_y_2d_arr/(h)

    el1, Cl_kk_dw0 = np.loadtxt('Cl_kk_dw0.dat')

    el2, Cl_kk_dwa = np.loadtxt('Cl_kk_dwa.dat')

    Cl_kk_dw0_f = interp1d(el1, Cl_kk_dw0, bounds_error=False,fill_value=0.0)

    Cl_kk_dwa_f = interp1d(el2, Cl_kk_dwa, bounds_error=False,fill_value=0.0)

    dC_ell_y_dw0 = Cl_kk_dw0_f(kperp_2d_arr*chi_A)

    dC_ell_y_dwa = Cl_kk_dwa_f(kperp_2d_arr*chi_A)


    return dC_ell_y_dsig8_zA, dC_ell_y_dOm_DE , dC_ell_y_dw0, dC_ell_y_dwa, dC_ell_y_dh


def bispec_distance_derivs_w0waCDM(ell_coarse_arr, y_coarse_arr, zbin_prop, fid_params, bias_var=1., gamma_var=1.):


    z_A = zbin_prop.z_A
    Deltanutilde = zbin_prop.Deltanutilde


	# comoving volume at z_max - comov vol at z_min * FOV solid angle / 4pi    --->>> cd.comoving_volume
    chi_A = zbin_prop.chi_A ; nutilde_A = zbin_prop.nutilde_A ; rnu_A = zbin_prop.rnu_A
    lambda_A = zbin_prop.lambda_A

    nu_tilde_min = zbin_prop.nu_tilde_min; nu_tilde_max = zbin_prop.nu_tilde_max
    z_A_min = zbin_prop.z_A_min ; z_A_max = zbin_prop.z_A_max

    chi_A_min = zbin_prop.chi_A_min; chi_A_max = zbin_prop.chi_A_max;
    delta_chipar_bin = zbin_prop.delta_chipar_bin

    FOV_A_sr = zbin_prop.FOV_A_sr
    delta_chiperp_bin = zbin_prop.delta_chiperp_bin
    apar = cosmological_parameters.apar ; aperp = cosmological_parameters.aperp


    qpar_min = zbin_prop.qpar_min ; qperp_min = zbin_prop.qperp_min
    num_qpar = 400 ; num_qperp = 399#4000,3999
    N_factor_integral_range=200
    qpar_arr = np.linspace(1e-7,N_factor_integral_range*qpar_min,num_qpar) ; qperp_arr = np.linspace(1e-7,N_factor_integral_range*qperp_min,num_qperp)


    K_21_qpar_arr = np.sin(qpar_arr*delta_chipar_bin/2.0)/ (qpar_arr*delta_chipar_bin/2.0)
    K_21_qperp_arr = np.sin(qperp_arr*delta_chiperp_bin/2.0)/(qperp_arr*delta_chiperp_bin/2.0)
    dK_21_qpar_arr = ( np.cos(qpar_arr*delta_chipar_bin/2.0)*qpar_arr*delta_chipar_bin/2.0 - np.sin(qpar_arr*delta_chipar_bin/2.0) )/(qpar_arr**2.*delta_chipar_bin/2.0)
    dK_21_qperp_arr = (np.cos(qperp_arr*delta_chiperp_bin/2.0)*qperp_arr*delta_chiperp_bin/2.0 - np.sin(qperp_arr*delta_chiperp_bin/2.0))/(qperp_arr**2.*delta_chiperp_bin/2.0)
    K_21_qpar_arr[0] = 1.0
    K_21_qperp_arr[0] = 1.0
    dK_21_qpar_arr[0] = 1.0
    dK_21_qperp_arr[0] = 1.0
    K_kappa_qperp_arr = K_21_qperp_arr
    K_kappa_qpar_arr = K_21_qpar_arr
    dK_kappa_qperp_arr = dK_21_qperp_arr
    dK_kappa_qpar_arr = dK_21_qpar_arr


    Int_alt_k_sinc_max = np.sin(np.amax(qperp_arr)*delta_chipar_bin/2.0)/ (np.amax(qperp_arr)*delta_chipar_bin/2.0)
    Int_alt_k_sinc_min = np.sin(qpar_min*delta_chipar_bin/2.0)/ (qpar_min*delta_chipar_bin/2.0)
    Int_alt_k_Pm_min = power_spec_functions.get_P_m_0(qpar_min)
    Int_alt_k_Pm_max = power_spec_functions.get_P_m_0(np.amax(qperp_arr))

    daperp_q_at_qmax = (aperp*np.amax(qperp_arr))**2. / (np.sqrt( np.amax(qperp_arr)**2. + np.amax(qpar_arr)**2.)  *aperp)
    dapar_q_at_qmax = (apar*np.amax(qpar_arr))**2. / (np.sqrt( np.amax(qperp_arr)**2. + np.amax(qpar_arr)**2.)  *apar)
    daperp_q_at_qmin = (aperp*qperp_arr[-1])**2. / (np.sqrt( qperp_arr[-1]**2. + qpar_arr[-1]**2.)  *aperp)
    dapar_q_at_qmin = (apar*qpar_arr[-1])**2. / (np.sqrt( qperp_arr[-1]**2. + qpar_arr[-1]**2.)  *apar)


    deriv_K_aperp = np.amax(qperp_arr)**2. * Int_alt_k_sinc_max**2. * Int_alt_k_Pm_max*daperp_q_at_qmax - qpar_min**2. *Int_alt_k_sinc_min**2. * Int_alt_k_Pm_min*daperp_q_at_qmin
    deriv_K_apar = np.amax(qperp_arr)**2. * Int_alt_k_sinc_max**2. * Int_alt_k_Pm_max*dapar_q_at_qmax - qpar_min**2. *Int_alt_k_sinc_min**2. * Int_alt_k_Pm_min*dapar_q_at_qmin

    K_21_qperp_2d_arr = np.outer(K_21_qperp_arr,np.ones(num_qpar))  ; K_21_qpar_2d_arr = np.outer(np.ones(num_qperp), K_21_qpar_arr)
    K_kappa_qperp_2d_arr = np.outer(K_kappa_qperp_arr,np.ones(num_qpar))  ; K_kappa_qpar_2d_arr = np.outer(np.ones(num_qperp), K_kappa_qpar_arr)

    dK_21_qperp_2d_arr = np.outer(dK_21_qperp_arr,np.ones(num_qpar))  ; dK_21_qpar_2d_arr = np.outer(np.ones(num_qperp), dK_21_qpar_arr)
    dK_kappa_qperp_2d_arr = np.outer(dK_kappa_qperp_arr,np.ones(num_qpar))  ; dK_kappa_qpar_2d_arr = np.outer(np.ones(num_qperp), dK_kappa_qpar_arr)


    q_par_2d_arr = np.outer(np.ones(num_qperp),qpar_arr)
    q_perp_2d_arr = np.outer(qperp_arr,np.ones(num_qpar))

    q_mag_2d_arr = np.sqrt(np.outer(qperp_arr,np.ones(num_qpar))**2 + np.outer(np.ones(num_qperp),qpar_arr)**2)
    fbao_q_2d_arr = power_spec_functions.get_fbao(q_mag_2d_arr)
    Pm_q_2d_arr = power_spec_functions.get_P_m_0(q_mag_2d_arr)

    Integral_ell_y_zA_over_qpar = np.trapz(0.5/np.pi**2 * Pm_q_2d_arr*K_21_qperp_2d_arr*K_kappa_qperp_2d_arr*K_21_qpar_2d_arr*K_kappa_qpar_2d_arr*q_perp_2d_arr, x=qperp_arr, axis=0)
    Integral_ell_y_zA_over_qpar_qperp = np.trapz(Integral_ell_y_zA_over_qpar, x=qpar_arr, axis=0)


    Soln_K_aperp = (1/2.*np.pi)**2. *  deriv_K_aperp/Integral_ell_y_zA_over_qpar_qperp
    Soln_K_apar = (1/2.*np.pi)**2. *  deriv_K_apar/Integral_ell_y_zA_over_qpar_qperp

    h=cosmological_parameters.h
    kP=np.loadtxt('Datafiles/P_k_nonlin_camb_z0.dat')  # goes up to k=18/Mpc
    P_spl_k=interp1d(kP[:,0]/h, kP[:,1]*h**3,bounds_error=False,fill_value=0.0)

    q_mag_arr_abs = qpar_arr
    dpk_dq = scipy.misc.derivative(power_spec_functions.get_P_m_0, q_mag_arr_abs,dx=1e-3)
    d_integrand = Pm_q_2d_arr*dK_21_qperp_2d_arr*dK_21_qpar_2d_arr*K_kappa_qperp_2d_arr*K_kappa_qpar_2d_arr\
                +Pm_q_2d_arr*K_21_qperp_2d_arr*K_21_qpar_2d_arr*dK_kappa_qperp_2d_arr*dK_kappa_qpar_2d_arr \
                + dpk_dq *K_21_qperp_2d_arr*K_21_qpar_2d_arr*K_kappa_qperp_2d_arr*K_kappa_qpar_2d_arr




    Integral_ell_y_zA_over_qpar_fbao = np.trapz(0.5/np.pi**2 * Pm_q_2d_arr * fbao_q_2d_arr/(1+fbao_q_2d_arr) *K_21_qperp_2d_arr*K_kappa_qperp_2d_arr*K_21_qpar_2d_arr*K_kappa_qpar_2d_arr*q_perp_2d_arr, x=qperp_arr, axis=0)
    Integral_ell_y_zA_over_qpar_qperp_fbao = np.trapz(Integral_ell_y_zA_over_qpar_fbao, x=qpar_arr, axis=0)


    Integral_ell_y_zA_coarse = Integral_ell_y_zA_over_qpar_qperp
    Integral_ell_y_zA_coarse_fbao = Integral_ell_y_zA_over_qpar_qperp_fbao

    C_ell_y_zA = np.zeros((ell_coarse_arr.size,y_coarse_arr.size))
    dPmdk_z0_ktot_2d_arr = np.zeros((ell_coarse_arr.size,y_coarse_arr.size))
    d2Pmdk_z0_ktot_2d_arr = np.zeros((ell_coarse_arr.size,y_coarse_arr.size))
    Pm_z0_ktot_2d_arr = np.zeros((ell_coarse_arr.size,y_coarse_arr.size))

    ell_coarse_min = np.min(ell_coarse_arr); ell_coarse_max = np.max(ell_coarse_arr)
    y_coarse_min = np.min(y_coarse_arr); y_coarse_max = np.max(y_coarse_arr)
    n_coarse_ell = ell_coarse_arr.size ; n_coarse_y = y_coarse_arr.size

   ### NOW COMPUTE 21-21-kappa BISPECTRUM

   # Set up 2d k arrays
    kperp_arr = ell_coarse_arr/chi_A; kpar_arr = y_coarse_arr/rnu_A
    ktot_2d_arr = np.sqrt(np.outer(ell_coarse_arr,np.ones(n_coarse_y))**2/chi_A**2 + np.outer(np.ones(n_coarse_ell),y_coarse_arr)**2/rnu_A**2)
    kpar_2d_arr = np.outer(np.ones(n_coarse_ell),kpar_arr)
    kperp_2d_arr = np.outer(kperp_arr,np.ones(n_coarse_y))
    mu_k_2d_arr = kpar_2d_arr/ktot_2d_arr
    fbao_2d_k_arr = power_spec_functions.get_fbao(ktot_2d_arr)

   # GET P_m DERIVATIVE
    k_arr = kpar_arr   # USE kpar_arr to get derivative since y_arr gives smaller kmin and larger kmax
    Pm_z0_karr=power_spec_functions.get_P_m_0(k_arr)
    dPmdk_z0_arr=scipy.misc.derivative(power_spec_functions.get_P_m_0, k_arr,dx=1e-3)
    dPmdk_z0_arr_interp = scipy.interpolate.interp1d(dPmdk_z0_arr,k_arr,bounds_error=False,fill_value=0.0)
    d2Pmdk_z0_arr=scipy.misc.derivative(dPmdk_z0_arr_interp, k_arr,dx=1e-3)
    dfbao_dk = scipy.misc.derivative(power_spec_functions.get_fbao, k_arr,dx=1e-3)

   # Now evaluate all terms in bispectrum

    Pm_z0_ktot_2d_arr[:,] = np.interp(ktot_2d_arr[:,],k_arr,Pm_z0_karr)
    dPmdk_z0_ktot_2d_arr[:,]= np.interp(ktot_2d_arr[:,],k_arr,dPmdk_z0_arr) # mtx_sq[:,]=np.interp(mtx[:,],x,y)
    d2Pmdk_z0_ktot_2d_arr[:,] =np.interp(ktot_2d_arr[:,],k_arr,d2Pmdk_z0_arr)

    F_bias_rsd_2d_arr = (power_spec_functions.get_HI_bias(z_A, bias_var)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)

    ##***********************************************##
    ## *** Updated Integral for Oliver kernel *****
    ##***********************************************##

    b2 = power_spec_functions.get_HI_bias_2nd_order(z_A)
    f = power_spec_functions.get_growth_factor_f(z_A, gamma_var)
    mk = mu_k_2d_arr
    b1 = power_spec_functions.get_HI_bias(z_A, bias_var)
    Z1 = (b1 + f*mk**2.)

    Updated_derivative_term = (1/3.)*(f*mk**2. - mk**2. + 2.)*(3.0 - dPmdk_z0_ktot_2d_arr*ktot_2d_arr/Pm_z0_ktot_2d_arr)
    Updated_derivative_independent_term = (1/(14*Z1))*(14*b1*f*mk**2. + 14*b1*(f/3.) + 26.*b1*(mk**2./3.) + 26.*(b1/3.) + 28*b2 + 14*f**2.*mk**4. - 14*f**2*mk**2  \
    -6*f*mk**4. + 38*f*(mk**2/3.))

    Kernel_2d_arr_updated = Updated_derivative_term + Updated_derivative_independent_term


    Kernel_2d_arr = Kernel_2d_arr_updated


    P_21_z0_2d_arr = F_bias_rsd_2d_arr**2 *power_spec_functions.get_P_m_0(ktot_2d_arr) * power_spec_functions.get_mean_temp(z_A)**2

    volume_A_max = cd.comoving_volume(z_A_max, **cosmo); volume_A_min = cd.comoving_volume(z_A_min, **cosmo)
    volume_A = (volume_A_max - volume_A_min) *FOV_A_sr/(4.0*np.pi)  #  # - CHECK THIS


    W_kappa_over_chi_sq=lensing_kernel.get_lensing_kernel_real_sp(chi_A) / chi_A**2

    Volume_lensing_kernel_growth_factor = W_kappa_over_chi_sq * power_spec_functions.get_growth_function_D(z_A)**4 * volume_A / (chi_A**2 * rnu_A)

    C_ell_y_zA =  Integral_ell_y_zA_coarse* P_21_z0_2d_arr * Volume_lensing_kernel_growth_factor * Kernel_2d_arr
    C_ell_y_zA_fbao =  Integral_ell_y_zA_coarse_fbao* P_21_z0_2d_arr * Volume_lensing_kernel_growth_factor * Kernel_2d_arr

    f_zA = power_spec_functions.get_growth_factor_f(z_A, bias_var)
    drsd_du2 = 2.*f_zA/ F_bias_rsd_2d_arr

    dlogpk_dk = Fpm.logpk_derivative(P_spl_k, ktot_2d_arr) # Numerical deriv.
    daperp_u2 = -2. * (kperp_2d_arr/kpar_2d_arr * aperp/apar * mu_k_2d_arr**2.)**2. / aperp
    dapar_u2 =   2. * (kperp_2d_arr/kpar_2d_arr* aperp/apar * mu_k_2d_arr**2.)**2. / apar
    daperp_k = (aperp*kperp_2d_arr)**2. / (ktot_2d_arr*aperp)
    daperp_q = (aperp*q_perp_2d_arr)**2. / (q_mag_2d_arr*aperp)
    dapar_k  = (apar*kpar_2d_arr)**2. / (ktot_2d_arr*apar)
    dapar_q = (apar*q_par_2d_arr)**2. / (q_mag_2d_arr*apar)
    dKernel_term_1=4.*power_spec_functions.get_growth_factor_f(z_A, gamma_var)*(1+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)
    dKernel_term_2 = 2*power_spec_functions.get_growth_factor_f(z_A, gamma_var)**2. * (2*mu_k_2d_arr**2-1.0)
    dKernel_term_3 = power_spec_functions.get_growth_factor_f(z_A, gamma_var)*(power_spec_functions.get_HI_bias_2nd_order(z_A)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*(2*mu_k_2d_arr**2-1.0)*2*(1+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2))
    dKernel_term_4 = 0.5*power_spec_functions.get_growth_factor_f(z_A, gamma_var)*(3.0 - dPmdk_z0_ktot_2d_arr*ktot_2d_arr/Pm_z0_ktot_2d_arr)
    dkernel_2d_arr = (((dKernel_term_1 + dKernel_term_2)*F_bias_rsd_2d_arr - dKernel_term_3)/F_bias_rsd_2d_arr**2.) + dKernel_term_4
    dlogPm_dlogk = d2Pmdk_z0_ktot_2d_arr * (ktot_2d_arr/Pm_z0_ktot_2d_arr) + dPmdk_z0_ktot_2d_arr*(Pm_z0_ktot_2d_arr-ktot_2d_arr*dPmdk_z0_ktot_2d_arr)/Pm_z0_ktot_2d_arr**2.
    ### For nonlinear cases sigma_nl must add extra terms dlogpk_dk + drsd_dk + dbias_k below as in Bull et al.
    dIntegral_ell_y_zA_over_qpar_aperp = np.trapz(0.5/np.pi**2 * d_integrand *q_perp_2d_arr*daperp_q, x=qperp_arr, axis=0)
    dIntegral_ell_y_zA_over_qpar_qperp_aperp = np.trapz(dIntegral_ell_y_zA_over_qpar_aperp, x=qpar_arr, axis=0)
    dIntegral_ell_y_zA_alternative = 2.*np.pi**2. * np.amax(Pm_q_2d_arr*K_21_qperp_2d_arr*K_kappa_qperp_2d_arr*K_21_qpar_2d_arr*K_kappa_qpar_2d_arr*q_perp_2d_arr*dapar_q)

    dIntegral_ell_y_zA_over_qpar_apar= np.trapz(0.5/np.pi**2 * d_integrand *q_perp_2d_arr*dapar_q, x=qperp_arr, axis=0)
    dIntegral_ell_y_zA_over_qpar_qperp_apar = np.trapz(dIntegral_ell_y_zA_over_qpar_apar, x=qpar_arr, axis=0)
    d2Pm_dk_aperp_apar = d2Pmdk_z0_ktot_2d_arr*(ktot_2d_arr/Pm_z0_ktot_2d_arr) + dPmdk_z0_ktot_2d_arr*(Pm_z0_ktot_2d_arr-ktot_2d_arr*dPmdk_z0_ktot_2d_arr)/(Pm_z0_ktot_2d_arr**2.)

    ##***********************************************##
    ## *** Updated Derivatives for kernel *****
    ##***********************************************##

    d_derivative_term_dmu = (3.0 - dPmdk_z0_ktot_2d_arr*ktot_2d_arr/Pm_z0_ktot_2d_arr)*(2.*f*mk/3. - 2.*mk/3.)
    d_derivative_independent_term_dmu =  (1./(Z1*14.))*(-b1*(-14.*f*mk - 26.*mk/3.) + b1*(14.*f*mk + 26.*mk/3.) - f*mk*(-21.*f*mk**2. + 7.*f + 9.*mk**2. - 19./3.) + f*mk*(21.*f*mk**2. - 7.*f - 9.*mk**2. + 19./3.) - f*(-7.*f*mk**3. + 7.*f*mk + 3.*mk**3. - 19.*mk/3.) + f*(7.*f*mk**3. - 7.*f*mk - 3.*mk**3. + 19.*mk/3.))
    dkernel_2d_arr_dmu = 2*mk*(d_derivative_term_dmu  + d_derivative_independent_term_dmu )


    deriv_P21_aperp_over_P21 = ( (2./aperp) + drsd_du2 * daperp_u2 \
                        + (dlogpk_dk)*daperp_k )
    deriv_P21_apar_over_P21 =  ( (1./apar)  + drsd_du2 * dapar_u2 \
                        + (dlogpk_dk)*dapar_k  )

    deriv_aperp = ( (4./aperp) + deriv_P21_aperp_over_P21  + Soln_K_aperp\
        +(dkernel_2d_arr_dmu*daperp_u2-dlogPm_dlogk*0.5*(1+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)*daperp_k )\
			/Kernel_2d_arr ) * C_ell_y_zA #+ dIntegral_ell_y_zA_over_qpar_qperp_aperp/Integral_ell_y_zA_coarse
    deriv_apar =   ( (1./apar) + deriv_P21_apar_over_P21  + Soln_K_apar\
        +(dkernel_2d_arr_dmu*dapar_u2-dlogPm_dlogk *0.5*(1+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)*dapar_k )\
			/Kernel_2d_arr ) * C_ell_y_zA #+ dIntegral_ell_y_zA_over_qpar_qperp_apar/Integral_ell_y_zA_coarse


    Abao_zA,sig8, b1_zA, b2_zA, f_zA, aperp, apar= fid_params
    sig8_zA = sig8*power_spec_functions.get_growth_function_D(z_A)


    dB_ell_y_dAbao = C_ell_y_zA*(fbao_2d_k_arr/(1.+fbao_2d_k_arr) + Integral_ell_y_zA_coarse_fbao/Integral_ell_y_zA_coarse \
							-0.5*(1+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)\
							*(ktot_2d_arr/ (1+power_spec_functions.get_growth_factor_f(z_A, gamma_var))**2. )*dfbao_dk  )

    sig8_deriv_kernel = 4.0 * Kernel_2d_arr / (sig8_zA*power_spec_functions.get_growth_function_D(z_A)) 	# TESTING sig8 INFO
    dB_ell_y_dsig8_zA = C_ell_y_zA * sig8_deriv_kernel/Kernel_2d_arr

    dB_ell_y_db2_zA = (2/Z1)* C_ell_y_zA/Kernel_2d_arr

    dB_ell_y_db1_zA = ((14.*f*mk**2. + 14.*f/3. + 26.*mk**2./3. + 26./3.)/14./(14.*Z1))* C_ell_y_zA/Kernel_2d_arr

    f_deriv_kernel =(1/(14.*Z1))*(-b1*(-7.*mk**2. - 7./3.) + b1*(7.*mk**2. + 7./3.) - f*mk*(-7.*mk**3. + 7.*mk) + f*mk*(7.*mk**3. - 7.*mk) - mk*(-7.*f*mk**3. + 7.*f*mk + 3.*mk**3. - 19.*mk/3.) + mk*(7.*f*mk**3. - 7.*f*mk - 3.*mk**3. + 19.*mk/3.))
    dB_ell_y_df_zA =  C_ell_y_zA*(2*mu_k_2d_arr**2/F_bias_rsd_2d_arr + f_deriv_kernel/Kernel_2d_arr)

    return dB_ell_y_dAbao,dB_ell_y_dsig8_zA, dB_ell_y_db1_zA,dB_ell_y_db2_zA , dB_ell_y_df_zA, deriv_aperp, deriv_apar


def get_three_param_FisherMatrix_Cl_21_21_auto_w0waCDM(zbin_prop, num_k_bins_ell = 15, num_k_bins_y=14):

	## SET Z BIN PROPERTIES

    Deltanutilde = zbin_prop.Deltanutilde    # Bin width
    deltanutilde = zbin_prop.deltanutilde	 # Channel width

    z_A = zbin_prop.z_A

    nutilde_A = zbin_prop.nutilde_A
    chi_A=zbin_prop.chi_A
    rnu_A=zbin_prop.rnu_A
    lambda_A = zbin_prop.lambda_A
    FOV_A_sr = zbin_prop.FOV_A_sr
    theta_min_sr = zbin_prop.theta_min_sr

	### SET RADIAL MODES

    y_min = zbin_prop.y_min_expt ; y_max = zbin_prop.y_max_expt
    result = zbin_prop.get_y_arr_expt() ; n_y = result[0]  ; y_arr = result[1]

	### SET TRANSVERSE MODES

    ell_min = zbin_prop.ell_min_expt  ; ell_max = zbin_prop.ell_max_expt
    result = zbin_prop.get_ell_arr_expt() ; n_ell = result[0]  ; ell_arr = result[1]   # CHANGE THESE TO _expt

    N_patches = zbin_prop.N_patches
    Mode_Volume_Factor = zbin_prop.Mode_Volume_Factor

	### COMPUTE AUTO-21-21 Fisher matrix for Tb, f, b1,b2 - MERGE ALL FISHER MATRIX CALCS INTO ONE METHOD


    aperp = cosmological_parameters.aperp
    apar = cosmological_parameters.apar
    sig8 = cosmological_parameters.sig8
    Abao = cosmological_parameters.Abao
    f_nl = cosmological_parameters.f_nl

    ell_min_ltd = ell_min ; y_min_ltd = y_min

    non_linear_scale = zbin_prop.knl
    ell_max_ltd = non_linear_scale*chi_A
    y_max_ltd = non_linear_scale*rnu_A

    ell_arr_ltd = np.linspace(ell_min_ltd,ell_max_ltd,n_ell); y_arr_ltd = np.linspace(y_min_ltd,y_max_ltd,n_y)

    f_zA = power_spec_functions.get_growth_factor_f(z_A, gamma_var=1.0)
    b1_zA=power_spec_functions.get_HI_bias(z_A, bias_var=1.0); b2_zA = power_spec_functions.get_HI_bias_2nd_order(z_A)


    fid_fisher_params = np.array([Abao,sig8, b1_zA, b2_zA,f_zA, aperp, apar])
    n_fisher_params = fid_fisher_params.size

    Cl_Abao,Cl_sig8, Cl_b1, Cl_b2, Cl_f, Cl_aperp, Cl_apar = Cl21_Distance_derivs_w0waCDM(zbin_prop, ell_arr_ltd, y_arr_ltd, fid_fisher_params, bias_var=1., gamma_var=1.)

    Cl_signal = Fpm.Cl_21_auto_ell_y_zA(ell_arr_ltd, y_arr_ltd, zbin_prop, bias_var=1., gamma_var=1.)

    parameter_deriv_C_ell_y_arr = np.array([Cl_Abao,Cl_sig8, Cl_b1, Cl_b2, Cl_f, Cl_aperp, Cl_apar])


    ell_2d_arr_ltd=np.outer(ell_arr_ltd,np.ones(n_y))

    Cl_21_noise_ell_arr=Fpm.HI_angular_noise_ell_y(ell_arr_ltd, zbin_prop, experiment_name='hirax', mode='interferometer', show_plots=False)
    Cl_21_noise_ell_y_2d_arr=  np.outer(Cl_21_noise_ell_arr,np.ones(n_y))
    Cl_21_noise_ell_y_2d_arr_allsky = Fpm.HI_angular_noise_ell_y_allsky(Cl_21_noise_ell_y_2d_arr)

    Cl_21_auto_ell_y_2d_arr=Fpm.Cl_21_auto_ell_y_zA(ell_arr_ltd, y_arr_ltd, zbin_prop, bias_var=1., gamma_var=1.) # 21cm angular PS at z vs ell,y

    variance_arr = (Cl_21_noise_ell_y_2d_arr_allsky + Cl_21_auto_ell_y_2d_arr)**2


    fisher_arr_zA_21_21 = np.zeros((n_fisher_params,n_fisher_params)) ; corr_arr_zA_21_21 = np.zeros((n_fisher_params,n_fisher_params))
    #wedge_factor = cd.hubble_z(z_A, **cosmo)*chi_A/(C*(1+z_A))
    wedge_factor_2 = cosmological_parameters.H0*cd.e_z(z_A, **cosmo)*chi_A/(3e5*(1+z_A))
    k1_w, k2_w = np.meshgrid(ell_arr_ltd, y_arr_ltd)
    #print('len ell', len(ell_arr_ltd))
    wedge_mask = k2_w >= k1_w/wedge_factor_2
    for ii, pp_i in enumerate(fid_fisher_params):
		deriv_arr_ii = parameter_deriv_C_ell_y_arr[ii,]
		for jj, pp_j in enumerate(fid_fisher_params):
			deriv_arr_jj = parameter_deriv_C_ell_y_arr[jj,]
			signal_arr_sq = deriv_arr_ii * deriv_arr_jj
			fisher_ratio_arr = signal_arr_sq/variance_arr
			fisher_ratio_arr  = np.where(wedge_mask.T, fisher_ratio_arr, 0.0)
			#plt.pcolormesh(fisher_ratio_arr)
			#plt.show()
			fisher_arr_zA_21_21[ii,jj] = 0.5 * Mode_Volume_Factor/(2*np.pi**2) * np.trapz(np.trapz(fisher_ratio_arr*ell_2d_arr_ltd, ell_arr_ltd, axis=0), y_arr_ltd, axis=0)


    fid_param_label = ['Abao','sig8','b1','b2','f', 'aperp', 'apar']

    return fid_param_label, fid_fisher_params, fisher_arr_zA_21_21


def get_Clkappa_distance_params_fisher_w0waCDM(zbin_prop, num_k_bins_ell = 15, num_k_bins_y=14):

	## SET Z BIN PROPERTIES

    Deltanutilde = zbin_prop.Deltanutilde    # Bin width
    deltanutilde = zbin_prop.deltanutilde	 # Channel width

    z_A = zbin_prop.z_A

    nutilde_A = zbin_prop.nutilde_A
    chi_A=zbin_prop.chi_A
    rnu_A=zbin_prop.rnu_A
    lambda_A = zbin_prop.lambda_A
    FOV_A_sr = zbin_prop.FOV_A_sr
    theta_min_sr = zbin_prop.theta_min_sr

    aperp = cosmological_parameters.aperp
    apar = cosmological_parameters.apar
    sig8 = cosmological_parameters.sig8
    Abao = cosmological_parameters.Abao


	### SET RADIAL MODES

    y_min = zbin_prop.y_min_expt ; y_max = zbin_prop.y_max_expt
    result = zbin_prop.get_y_arr_expt() ; n_y = result[0]  ; y_arr = result[1]

	### SET TRANSVERSE MODES

    ell_min = zbin_prop.ell_min_expt  ; ell_max = zbin_prop.ell_max_expt
    result = zbin_prop.get_ell_arr_expt() ; n_ell = result[0]  ; ell_arr = result[1]   # CHANGE THESE TO _expt

    N_patches = zbin_prop.N_patches
    Mode_Volume_Factor = zbin_prop.Mode_Volume_Factor

    ### COMPUTE AUTO-21-21 Fisher matrix for Tb, f, b1,b2 - MERGE ALL FISHER MATRIX CALCS INTO ONE METHOD

    sig8 = cosmological_parameters.sig8
    h = cosmological_parameters.h
    Om_m = cosmological_parameters.om_m
    Om_DE = cosmological_parameters.om_L
    w0 = cosmological_parameters.w0
    wa = cosmological_parameters.wa


    ell_min_ltd = ell_min ; y_min_ltd = y_min
    non_linear_scale = zbin_prop.knl
    ell_max_ltd = non_linear_scale*chi_A
    y_max_ltd = non_linear_scale*rnu_A
    ell_arr_ltd = np.linspace(ell_min_ltd,ell_max_ltd,n_ell); y_arr_ltd = np.linspace(y_min_ltd,y_max_ltd,n_y)

    Omega_HI_zA=power_spec_functions.get_Omega_HI(z_A) ; f_zA = power_spec_functions.get_growth_factor_f(z_A, gamma_var=1.0)
    b1_zA=power_spec_functions.get_HI_bias(z_A, bias_var=1.0); b2_zA = power_spec_functions.get_HI_bias_2nd_order(z_A)

    fid_fisher_params = np.array([sig8, Om_DE, w0, wa, h]) #, Omega_HI_zA	 # T_b(z_i), sig8, f, b1, b2   // T_b(z_i) in place of Omega_HI(z_i)
    n_fisher_params = fid_fisher_params.size

    Cl_sig8, Cl_Om_DE, Cl_w0, Cl_wa, Cl_h = Cl_kappa_distance_derivs_w0waCDM(ell_arr_ltd, y_arr_ltd, zbin_prop, fid_fisher_params, bias_var=1., gamma_var=1.)

    Cl_signal = Fpm.CMB_convergence_power(ell_arr_ltd)

    parameter_deriv_C_ell_y_arr = np.array([Cl_sig8, Cl_Om_DE, Cl_w0, Cl_wa, Cl_h])  #,Cl_Omega_HI

    ell_2d_arr_ltd=np.outer(ell_arr_ltd,np.ones(n_y))

    Cl_kappa_noise_ell_arr=Fpm.CMB_convergence_noise(ell_arr_ltd)
    Cl_kappa_noise_ell_y_2d_arr=  np.outer(Cl_kappa_noise_ell_arr,np.ones(n_y))
    Cl_kappa_auto_ell_y_2d_arr= np.outer(Cl_signal,np.ones(n_y))

    variance_arr = (Cl_kappa_noise_ell_y_2d_arr+ Cl_kappa_auto_ell_y_2d_arr)**2.

    fisher_arr_zA_kappa = np.zeros((n_fisher_params,n_fisher_params)) ; corr_arr_zA_kappa = np.zeros((n_fisher_params,n_fisher_params))

    for ii, pp_i in enumerate(fid_fisher_params):
		deriv_arr_ii = parameter_deriv_C_ell_y_arr[ii,]
		for jj, pp_j in enumerate(fid_fisher_params):
			deriv_arr_jj = parameter_deriv_C_ell_y_arr[jj,]
			signal_arr_sq = deriv_arr_ii * deriv_arr_jj
			fisher_ratio_arr = signal_arr_sq/variance_arr
			fisher_arr_zA_kappa[ii,jj] = 0.5 * Mode_Volume_Factor/(2.*np.pi**2) * np.trapz(np.trapz(fisher_ratio_arr*ell_2d_arr_ltd, ell_arr_ltd, axis=0), y_arr_ltd, axis=0)

    fid_param_label = ['sig8', 'Om_DE', 'w0', 'wa', 'h'] #, 'Omega_HI'

    return fid_param_label, fid_fisher_params, fisher_arr_zA_kappa


def get_distance_params_bispec_fisher_w0waCDM(zbin_prop, num_k_bins_ell = 15, num_k_bins_y=14):

	## SET Z BIN PROPERTIES

    Deltanutilde = zbin_prop.Deltanutilde    # Bin width
    deltanutilde = zbin_prop.deltanutilde	 # Channel width

    z_A = zbin_prop.z_A

    nutilde_A = zbin_prop.nutilde_A
    chi_A=zbin_prop.chi_A
    rnu_A=zbin_prop.rnu_A
    lambda_A = zbin_prop.lambda_A
    FOV_A_sr = zbin_prop.FOV_A_sr
    theta_min_sr = zbin_prop.theta_min_sr

    aperp = cosmological_parameters.aperp
    apar = cosmological_parameters.apar
    sig8 = cosmological_parameters.sig8
    Abao = cosmological_parameters.Abao
    fnl = cosmological_parameters.f_nl


	### SET RADIAL MODES

    y_min = zbin_prop.y_min_expt ; y_max = zbin_prop.y_max_expt
    result = zbin_prop.get_y_arr_expt() ; n_y = result[0]  ; y_arr = result[1]

	### SET TRANSVERSE MODES

    ell_min = zbin_prop.ell_min_expt  ; ell_max = zbin_prop.ell_max_expt
    result = zbin_prop.get_ell_arr_expt() ; n_ell = result[0]  ; ell_arr = result[1]   # CHANGE THESE TO _expt

    N_patches = zbin_prop.N_patches
    Mode_Volume_Factor = zbin_prop.Mode_Volume_Factor

	### GET 21-21 Auto Power Spectrum - signal and noise
    Cl_21_auto_ell_y_2d_arr=Fpm.Cl_21_auto_ell_y_zA(ell_arr, y_arr, zbin_prop, bias_var=1., gamma_var=1.) # 21cm angular PS at z vs ell,y
    Cl_21_noise_ell_arr=Fpm.HI_angular_noise_ell_y(ell_arr, zbin_prop, experiment_name='hirax', mode='interferometer', show_plots=False)
    Cl_21_noise_ell_y_2d_arr=  np.outer(Cl_21_noise_ell_arr,np.ones(n_y))

    CMB_kappa_average_variance_sky = Fpm.compute_average_kappa_variance_sky_zA(zbin_prop, bias_var=1., gamma_var=1.)


    ell_min_ltd = ell_min ; y_min_ltd = y_min
    non_linear_scale = zbin_prop.knl
    ell_max_ltd = non_linear_scale*chi_A
    y_max_ltd = non_linear_scale*rnu_A
    ell_arr_ltd = np.linspace(ell_min_ltd,ell_max_ltd,n_ell); y_arr_ltd = np.linspace(y_min_ltd,y_max_ltd,n_y)

    Omega_HI_zA=power_spec_functions.get_Omega_HI(z_A) ; f_zA = power_spec_functions.get_growth_factor_f(z_A, gamma_var=1.0)
    b1_zA=power_spec_functions.get_HI_bias(z_A, bias_var=1.0); b2_zA = power_spec_functions.get_HI_bias_2nd_order(z_A)

    fid_fisher_params = np.array([Abao,sig8, b1_zA, b2_zA, f_zA, aperp, apar]) #, Omega_HI_zA	 # T_b(z_i), sig8, f, b1, b2   // T_b(z_i) in place of Omega_HI(z_i)
    n_fisher_params = fid_fisher_params.size

    print "STARTING FISHER MATRIX"

    Bl_Abao,Bl_sig8,Bl_b1,Bl_b2 ,Bl_f,Bl_aperp, Bl_apar =  bispec_distance_derivs_w0waCDM(ell_arr_ltd, y_arr_ltd, zbin_prop, fid_fisher_params, bias_var=1., gamma_var=1.)

    Bl_signal = Fpm.integ_bispec_kappa_21_21_ell_y_zA(ell_arr_ltd, y_arr_ltd, zbin_prop, bias_var=1., gamma_var=1.)


    parameter_deriv_ell_y_arr = np.array([Bl_Abao,Bl_sig8,Bl_b1,Bl_b2,Bl_f,Bl_aperp, Bl_apar])#,Bl_Omega_HI # parameter_deriv_ell_y_arr[ii,]

    ell_2d_arr_ltd=np.outer(ell_arr_ltd,np.ones(n_y))
    CMB_kappa_noise_ell = Fpm.CMB_convergence_noise(ell_2d_arr_ltd)
    CMB_kappa_signal_ell = Fpm.CMB_convergence_power(ell_2d_arr_ltd)

    Cl_21_noise_ell_arr=Fpm.HI_angular_noise_ell_y(ell_arr_ltd, zbin_prop, experiment_name='hirax', mode='interferometer', show_plots=False)
    Cl_21_noise_ell_y_2d_arr=  np.outer(Cl_21_noise_ell_arr,np.ones(n_y))
    Cl_21_noise_ell_y_2d_arr_allsky = Cl_21_noise_ell_y_2d_arr/fsky

    variance_arr = 6.* N_patches**2. * Bl_signal**2. + \
    (3.*fsky**2.)*(Cl_21_noise_ell_y_2d_arr_allsky + Cl_21_auto_ell_y_2d_arr)**2. * CMB_kappa_average_variance_sky # (CMB_kappa_noise_ell+CMB_kappa_signal_ell)
    fisher_arr_zA_21_21_kappa = np.zeros((n_fisher_params,n_fisher_params)) ; corr_arr_zA=np.zeros((n_fisher_params,n_fisher_params))
    variance_arr = variance_arr
    C = 3e5*3.24e-23
    wedge_factor = cd.hubble_z(z_A, **cosmo)*chi_A/(C*(1+z_A))
    wedge_factor_2 = cosmological_parameters.H0*cd.e_z(z_A, **cosmo)*chi_A/(3e5*(1+z_A))
    k1_w, k2_w = np.meshgrid(ell_arr_ltd, y_arr_ltd)
    #print('len ell', len(ell_arr_ltd))
    wedge_mask = k2_w >= k1_w/wedge_factor_2
    #print('wedge_mask',wedge_mask)

    for ii, pp_i in enumerate(fid_fisher_params):
		deriv_arr_ii = parameter_deriv_ell_y_arr[ii,]
		for jj, pp_j in enumerate(fid_fisher_params):
			deriv_arr_jj = parameter_deriv_ell_y_arr[jj,]
			signal_arr_sq = deriv_arr_ii * deriv_arr_jj
			fisher_ratio_arr = N_patches**2 * signal_arr_sq/variance_arr
			fisher_ratio_arr  = np.where(wedge_mask.T, fisher_ratio_arr, 0.0)
			#print(cut)
			#plt.pcolormesh(fisher_ratio_arr*ell_2d_arr_ltd)
			#plt.colorbar()
			#plt.show()
			fisher_arr_zA_21_21_kappa[ii,jj] = 0.5 * Mode_Volume_Factor/(2*np.pi**2) * np.trapz(np.trapz(fisher_ratio_arr*ell_2d_arr_ltd, ell_arr_ltd, axis=0), y_arr_ltd, axis=0)


    fid_param_label = ['Abao','sig8', 'b1', 'b2', 'f', 'aperp', 'apar'] #,'Omega_HI


    return fid_param_label, fid_fisher_params, fisher_arr_zA_21_21_kappa


##############################################################################################################
# Below are derivatives and Fisher matrices for neutrino mass-w0waCDM
##############################################################################################################



def Cl21_Distance_derivs_Neutrino_mass(zbin_prop,ell_arr, y_arr,fid_params, bias_var=1., gamma_var=1.):

   n_ell = ell_arr.size ; n_y = y_arr.size
   z_A = zbin_prop.z_A
   chi_A = zbin_prop.chi_A ; nutilde_A = zbin_prop.nutilde_A ; rnu_A = zbin_prop.rnu_A
   kperp_arr = ell_arr/chi_A; kpar_arr = y_arr/rnu_A
   apar = cosmological_parameters.apar ; aperp = cosmological_parameters.aperp

   ktot_2d_arr = np.sqrt(np.outer(ell_arr,np.ones(n_y))**2/chi_A**2 + np.outer(np.ones(n_ell),y_arr)**2/rnu_A**2)
   kpar_2d_arr = np.outer(np.ones(n_ell),kpar_arr)
   kperp_2d_arr = np.outer(kperp_arr,np.ones(n_y))
   mu_k_2d_arr = kpar_2d_arr/ktot_2d_arr
   fbao_2d_arr = power_spec_functions.get_fbao(ktot_2d_arr)
   alpha_fnl_2d_arr = 3e5**2.*2.*ktot_2d_arr**2.* power_spec_functions.get_transfer_function(ktot_2d_arr)*power_spec_functions.get_growth_function_D(z_A)\
    				  /(3. * cosmological_parameters.om_m * (cosmological_parameters.H0)**2.)
   beta_fnl = 2.*cosmological_parameters.delta_c_fnl* ( power_spec_functions.get_HI_bias(z_A,bias_var) - 1.)


   F_bias_rsd_2d_arr = power_spec_functions.get_HI_bias(z_A, bias_var)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2
   F_bias_rsd_sq_2d_arr = F_bias_rsd_2d_arr**2
   F_bias_rsd_fnl_2d_arr = power_spec_functions.get_HI_bias(z_A, bias_var)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2 + \
   cosmological_parameters.f_nl * beta_fnl / alpha_fnl_2d_arr

   P_21_tot_2d_arr = F_bias_rsd_sq_2d_arr *power_spec_functions.get_P_m_0(ktot_2d_arr) * \
	power_spec_functions.get_mean_temp(z_A)**2 * power_spec_functions.get_growth_function_D(z_A)**2  # mK^2
   P_21_tot_2d_arr_fnl =  F_bias_rsd_fnl_2d_arr*power_spec_functions.get_P_m_0(ktot_2d_arr) * \
 	power_spec_functions.get_mean_temp(z_A)**2 * power_spec_functions.get_growth_function_D(z_A)**2

   Cl_21_auto_ell_y_2d_arr = P_21_tot_2d_arr / (chi_A**2 * rnu_A)
   Cl_21_auto_ell_y_2d_arr_fnl = P_21_tot_2d_arr_fnl / (chi_A**2 * rnu_A)

   f_zA = power_spec_functions.get_growth_factor_f(z_A, bias_var)
   drsd_du2 = 2.*f_zA/ F_bias_rsd_2d_arr

   h=cosmological_parameters.h
   kP=np.loadtxt('Datafiles/P_k_nonlin_camb_z0.dat')  # goes up to k=18/Mpc
   P_spl_k=interp1d(kP[:,0]/h, kP[:,1]*h**3,bounds_error=False,fill_value=0.0)


   dlogpk_dk = Fpm.logpk_derivative(P_spl_k, ktot_2d_arr) # Numerical deriv.
   daperp_u2 = -2. * (kperp_2d_arr/kpar_2d_arr * aperp/apar * mu_k_2d_arr**2.)**2. / aperp
   dapar_u2 =   2. * (kperp_2d_arr/kpar_2d_arr* aperp/apar * mu_k_2d_arr**2.)**2. / apar
   daperp_k = (aperp*kperp_2d_arr)**2. / (ktot_2d_arr*aperp)
   dapar_k  = (apar*kpar_2d_arr)**2. / (ktot_2d_arr*apar)

   ### For nonlinear cases sigma_nl must add extra terms dlogpk_dk + drsd_dk + dbias_k below as in Bull et al.
   deriv_aperp = ( (2./aperp) + drsd_du2 * daperp_u2 \
                       + (dlogpk_dk)*daperp_k ) * Cl_21_auto_ell_y_2d_arr
   deriv_apar =  ( (1./apar)  + drsd_du2 * dapar_u2 \
                       + (dlogpk_dk)*dapar_k  ) * Cl_21_auto_ell_y_2d_arr

   deriv_f = 2.*mu_k_2d_arr**2./ F_bias_rsd_2d_arr *  Cl_21_auto_ell_y_2d_arr

   Abao_zA,sig8, b1_zA, b2_zA, Mv_zA, f_zA, aperp, apar= fid_params
   sig8_zA = sig8*power_spec_functions.get_growth_function_D(z_A)

   dC_ell_y_dAbao = Cl_21_auto_ell_y_2d_arr*fbao_2d_arr/(1.+fbao_2d_arr)

   sig8_deriv_kernel = 2.0/(sig8_zA*power_spec_functions.get_growth_function_D(z_A))
   dC_ell_y_dsig8_zA = Cl_21_auto_ell_y_2d_arr * sig8_deriv_kernel

   b2_deriv_kernel = 0.0
   dC_ell_y_db2_zA = Cl_21_auto_ell_y_2d_arr * b2_deriv_kernel

   b1_deriv_kernel = 2.0/F_bias_rsd_2d_arr
   dC_ell_y_db1_zA = Cl_21_auto_ell_y_2d_arr*b1_deriv_kernel

   fnl_deriv_kernel = 2.0*beta_fnl/(alpha_fnl_2d_arr*F_bias_rsd_fnl_2d_arr)
   dC_ell_y_dfnl_zA = Cl_21_auto_ell_y_2d_arr*fnl_deriv_kernel

   #dC_ell_dMV = logpMv_derivative(ktot_2d_arr)*Cl_21_auto_ell_y_2d_arr

   dC_ell_dMV = (Fpm.logpMv_derivative_2(ktot_2d_arr)/P_spl_k(ktot_2d_arr))*Cl_21_auto_ell_y_2d_arr

   return  dC_ell_y_dAbao, dC_ell_y_dsig8_zA, dC_ell_y_db1_zA, dC_ell_y_db2_zA, dC_ell_dMV, deriv_f, deriv_aperp, deriv_apar


def Cl_kappa_distance_derivs_Neutrino_mass(ell_arr, y_arr, zbin_prop, fid_params, bias_var=1., gamma_var=1.): # 21cm angular PS at z vs ell,y

    n_ell = ell_arr.size ; n_y = y_arr.size
    z_A = zbin_prop.z_A
    chi_A = zbin_prop.chi_A ; nutilde_A = zbin_prop.nutilde_A ; rnu_A = zbin_prop.rnu_A
    chi_max = zbin_prop.chi_A_min; chi_min = zbin_prop.chi_A_max
    kperp_arr = ell_arr/chi_A; kpar_arr = y_arr/rnu_A
    apar = cosmological_parameters.apar ; aperp = cosmological_parameters.aperp


    ktot_2d_arr = np.sqrt(np.outer(ell_arr,np.ones(n_y))**2/chi_A**2 + np.outer(np.ones(n_ell),y_arr)**2/rnu_A**2)
    kpar_2d_arr = np.outer(np.ones(n_ell),kpar_arr)
    kperp_2d_arr = np.outer(kperp_arr,np.ones(n_y))
    fbao_2d_arr = power_spec_functions.get_fbao(ktot_2d_arr)


    Cl_kappa_auto_ell_y_2d_arr = Fpm.CMB_convergence_power(kperp_2d_arr*chi_A)


	# COMPUTE Cl_21 DERIVATIVES
    f_zA = power_spec_functions.get_growth_factor_f(z_A, bias_var)

    h=cosmological_parameters.h
    kP=np.loadtxt('Datafiles/P_k_nonlin_camb_z0.dat')  # goes up to k=18/Mpc
    P_spl_k=interp1d(kP[:,0]/h, kP[:,1]*h**3,bounds_error=False,fill_value=0.0)


    dlogpk_dk = Fpm.logpk_derivative(P_spl_k, ktot_2d_arr) # Numerical deriv.
    daperp_k = (aperp*kperp_2d_arr)**2. / (ktot_2d_arr*aperp)
    dapar_k  = (apar*kpar_2d_arr)**2. / (ktot_2d_arr*apar)

   ### For nonlinear cases sigma_nl must add extra terms dlogpk_dk + drsd_dk + dbias_k below as in Bull et al.
    W_kappa_over_chi_sq=lensing_kernel.get_lensing_kernel_real_sp(chi_A)**2.*power_spec_functions.get_growth_function_D(z_A)**2. / chi_A**2.


    sig8, Mv, Om_DE, h, w0, wa = fid_params
    sig8_zA = sig8
    Om_m = 1 - Om_DE


    sig8_deriv_kernel = 2.0/(sig8_zA)
    dC_ell_y_dsig8_zA = Cl_kappa_auto_ell_y_2d_arr * sig8_deriv_kernel

    dC_ell_y_dOm_m = 2.0*Cl_kappa_auto_ell_y_2d_arr/(Om_m)
    dC_ell_y_dOm_DE = -2.0*Cl_kappa_auto_ell_y_2d_arr/(Om_m)


    dC_ell_y_dh = 4.0*Cl_kappa_auto_ell_y_2d_arr/(h)

    el1, Cl_kk_dw0 = np.loadtxt('Cl_kk_dw0.dat')

    el2, Cl_kk_dwa = np.loadtxt('Cl_kk_dwa.dat')

    Cl_kk_dw0_f = interp1d(el1, Cl_kk_dw0, bounds_error=False,fill_value=0.0)

    Cl_kk_dwa_f = interp1d(el2, Cl_kk_dwa, bounds_error=False,fill_value=0.0)

    dC_ell_y_dw0 = Cl_kk_dw0_f(kperp_2d_arr*chi_A)

    dC_ell_y_dwa = Cl_kk_dwa_f(kperp_2d_arr*chi_A)

    el3, Cl_kk_dmnu = np.loadtxt('dClkk_dmu.dat')

    el4, Cl_kk_dmnu1 = np.loadtxt('dClkk_dmu_om_m_029.dat')

    Cl_kk_dmnu_f = interp1d(el3, Cl_kk_dmnu, bounds_error=False,fill_value=0.0)

    Cl_kk_dmnu_f1 = interp1d(el4, Cl_kk_dmnu1, bounds_error=False,fill_value=0.0)


    dC_ell_y_dmnu = Cl_kk_dmnu_f(kperp_2d_arr*chi_A)

    return dC_ell_y_dsig8_zA, dC_ell_y_dmnu, dC_ell_y_dOm_DE , dC_ell_y_dw0, dC_ell_y_dwa, dC_ell_y_dh




def bispec_distance_derivs_Neutrino_mass(ell_coarse_arr, y_coarse_arr, zbin_prop, fid_params, bias_var=1., gamma_var=1.):


    z_A = zbin_prop.z_A
    Deltanutilde = zbin_prop.Deltanutilde


	# comoving volume at z_max - comov vol at z_min * FOV solid angle / 4pi    --->>> cd.comoving_volume
    chi_A = zbin_prop.chi_A ; nutilde_A = zbin_prop.nutilde_A ; rnu_A = zbin_prop.rnu_A
    lambda_A = zbin_prop.lambda_A

    nu_tilde_min = zbin_prop.nu_tilde_min; nu_tilde_max = zbin_prop.nu_tilde_max
    z_A_min = zbin_prop.z_A_min ; z_A_max = zbin_prop.z_A_max

    chi_A_min = zbin_prop.chi_A_min; chi_A_max = zbin_prop.chi_A_max;
    delta_chipar_bin = zbin_prop.delta_chipar_bin

    FOV_A_sr = zbin_prop.FOV_A_sr
    delta_chiperp_bin = zbin_prop.delta_chiperp_bin
    apar = cosmological_parameters.apar ; aperp = cosmological_parameters.aperp


    qpar_min = zbin_prop.qpar_min ; qperp_min = zbin_prop.qperp_min
    num_qpar = 400 ; num_qperp = 399#4000,3999
    N_factor_integral_range=200
    qpar_arr = np.linspace(1e-7,N_factor_integral_range*qpar_min,num_qpar) ; qperp_arr = np.linspace(1e-7,N_factor_integral_range*qperp_min,num_qperp)


    K_21_qpar_arr = np.sin(qpar_arr*delta_chipar_bin/2.0)/ (qpar_arr*delta_chipar_bin/2.0)
    K_21_qperp_arr = np.sin(qperp_arr*delta_chiperp_bin/2.0)/(qperp_arr*delta_chiperp_bin/2.0)
    dK_21_qpar_arr = ( np.cos(qpar_arr*delta_chipar_bin/2.0)*qpar_arr*delta_chipar_bin/2.0 - np.sin(qpar_arr*delta_chipar_bin/2.0) )/(qpar_arr**2.*delta_chipar_bin/2.0)
    dK_21_qperp_arr = (np.cos(qperp_arr*delta_chiperp_bin/2.0)*qperp_arr*delta_chiperp_bin/2.0 - np.sin(qperp_arr*delta_chiperp_bin/2.0))/(qperp_arr**2.*delta_chiperp_bin/2.0)
    K_21_qpar_arr[0] = 1.0
    K_21_qperp_arr[0] = 1.0
    dK_21_qpar_arr[0] = 1.0
    dK_21_qperp_arr[0] = 1.0
    K_kappa_qperp_arr = K_21_qperp_arr
    K_kappa_qpar_arr = K_21_qpar_arr
    dK_kappa_qperp_arr = dK_21_qperp_arr
    dK_kappa_qpar_arr = dK_21_qpar_arr


    Int_alt_k_sinc_max = np.sin(np.amax(qperp_arr)*delta_chipar_bin/2.0)/ (np.amax(qperp_arr)*delta_chipar_bin/2.0)
    Int_alt_k_sinc_min = np.sin(qpar_min*delta_chipar_bin/2.0)/ (qpar_min*delta_chipar_bin/2.0)
    Int_alt_k_Pm_min = power_spec_functions.get_P_m_0(qpar_min)
    Int_alt_k_Pm_max = power_spec_functions.get_P_m_0(np.amax(qperp_arr))

    daperp_q_at_qmax = (aperp*np.amax(qperp_arr))**2. / (np.sqrt( np.amax(qperp_arr)**2. + np.amax(qpar_arr)**2.)  *aperp)
    dapar_q_at_qmax = (apar*np.amax(qpar_arr))**2. / (np.sqrt( np.amax(qperp_arr)**2. + np.amax(qpar_arr)**2.)  *apar)
    daperp_q_at_qmin = (aperp*qperp_arr[-1])**2. / (np.sqrt( qperp_arr[-1]**2. + qpar_arr[-1]**2.)  *aperp)
    dapar_q_at_qmin = (apar*qpar_arr[-1])**2. / (np.sqrt( qperp_arr[-1]**2. + qpar_arr[-1]**2.)  *apar)


    deriv_K_aperp = np.amax(qperp_arr)**2. * Int_alt_k_sinc_max**2. * Int_alt_k_Pm_max*daperp_q_at_qmax - qpar_min**2. *Int_alt_k_sinc_min**2. * Int_alt_k_Pm_min*daperp_q_at_qmin
    deriv_K_apar = np.amax(qperp_arr)**2. * Int_alt_k_sinc_max**2. * Int_alt_k_Pm_max*dapar_q_at_qmax - qpar_min**2. *Int_alt_k_sinc_min**2. * Int_alt_k_Pm_min*dapar_q_at_qmin

    K_21_qperp_2d_arr = np.outer(K_21_qperp_arr,np.ones(num_qpar))  ; K_21_qpar_2d_arr = np.outer(np.ones(num_qperp), K_21_qpar_arr)
    K_kappa_qperp_2d_arr = np.outer(K_kappa_qperp_arr,np.ones(num_qpar))  ; K_kappa_qpar_2d_arr = np.outer(np.ones(num_qperp), K_kappa_qpar_arr)

    dK_21_qperp_2d_arr = np.outer(dK_21_qperp_arr,np.ones(num_qpar))  ; dK_21_qpar_2d_arr = np.outer(np.ones(num_qperp), dK_21_qpar_arr)
    dK_kappa_qperp_2d_arr = np.outer(dK_kappa_qperp_arr,np.ones(num_qpar))  ; dK_kappa_qpar_2d_arr = np.outer(np.ones(num_qperp), dK_kappa_qpar_arr)


    q_par_2d_arr = np.outer(np.ones(num_qperp),qpar_arr)
    q_perp_2d_arr = np.outer(qperp_arr,np.ones(num_qpar))

    q_mag_2d_arr = np.sqrt(np.outer(qperp_arr,np.ones(num_qpar))**2 + np.outer(np.ones(num_qperp),qpar_arr)**2)
    fbao_q_2d_arr = power_spec_functions.get_fbao(q_mag_2d_arr)
    Pm_q_2d_arr = power_spec_functions.get_P_m_0(q_mag_2d_arr)

    Integral_ell_y_zA_over_qpar = np.trapz(0.5/np.pi**2 * Pm_q_2d_arr*K_21_qperp_2d_arr*K_kappa_qperp_2d_arr*K_21_qpar_2d_arr*K_kappa_qpar_2d_arr*q_perp_2d_arr, x=qperp_arr, axis=0)
    Integral_ell_y_zA_over_qpar_qperp = np.trapz(Integral_ell_y_zA_over_qpar, x=qpar_arr, axis=0)

    Integral_ell_y_zA_over_qpar_Mv = np.trapz(0.5/np.pi**2 * Fpm.logpMv_derivative_2(q_mag_2d_arr) *K_21_qperp_2d_arr*K_kappa_qperp_2d_arr*K_21_qpar_2d_arr*K_kappa_qpar_2d_arr*q_perp_2d_arr, x=qperp_arr, axis=0)
    Integral_ell_y_zA_over_qpar_qperp_Mv = np.trapz(Integral_ell_y_zA_over_qpar, x=qpar_arr, axis=0)


    Soln_K_aperp = (1/2.*np.pi)**2. *  deriv_K_aperp/Integral_ell_y_zA_over_qpar_qperp
    Soln_K_apar = (1/2.*np.pi)**2. *  deriv_K_apar/Integral_ell_y_zA_over_qpar_qperp

    h=cosmological_parameters.h
    kP=np.loadtxt('Datafiles/P_k_nonlin_camb_z0.dat')  # goes up to k=18/Mpc
    P_spl_k=interp1d(kP[:,0]/h, kP[:,1]*h**3,bounds_error=False,fill_value=0.0)

    q_mag_arr_abs = qpar_arr
    dpk_dq = scipy.misc.derivative(power_spec_functions.get_P_m_0, q_mag_arr_abs,dx=1e-3)
    d_integrand = Pm_q_2d_arr*dK_21_qperp_2d_arr*dK_21_qpar_2d_arr*K_kappa_qperp_2d_arr*K_kappa_qpar_2d_arr\
                +Pm_q_2d_arr*K_21_qperp_2d_arr*K_21_qpar_2d_arr*dK_kappa_qperp_2d_arr*dK_kappa_qpar_2d_arr \
                + dpk_dq *K_21_qperp_2d_arr*K_21_qpar_2d_arr*K_kappa_qperp_2d_arr*K_kappa_qpar_2d_arr




    Integral_ell_y_zA_over_qpar_fbao = np.trapz(0.5/np.pi**2 * Pm_q_2d_arr * fbao_q_2d_arr/(1+fbao_q_2d_arr) *K_21_qperp_2d_arr*K_kappa_qperp_2d_arr*K_21_qpar_2d_arr*K_kappa_qpar_2d_arr*q_perp_2d_arr, x=qperp_arr, axis=0)
    Integral_ell_y_zA_over_qpar_qperp_fbao = np.trapz(Integral_ell_y_zA_over_qpar_fbao, x=qpar_arr, axis=0)


    Integral_ell_y_zA_coarse = Integral_ell_y_zA_over_qpar_qperp
    Integral_ell_y_zA_coarse_fbao = Integral_ell_y_zA_over_qpar_qperp_fbao

    C_ell_y_zA = np.zeros((ell_coarse_arr.size,y_coarse_arr.size))
    dPmdk_z0_ktot_2d_arr = np.zeros((ell_coarse_arr.size,y_coarse_arr.size))
    d2Pmdk_z0_ktot_2d_arr = np.zeros((ell_coarse_arr.size,y_coarse_arr.size))
    Pm_z0_ktot_2d_arr = np.zeros((ell_coarse_arr.size,y_coarse_arr.size))

    ell_coarse_min = np.min(ell_coarse_arr); ell_coarse_max = np.max(ell_coarse_arr)
    y_coarse_min = np.min(y_coarse_arr); y_coarse_max = np.max(y_coarse_arr)
    n_coarse_ell = ell_coarse_arr.size ; n_coarse_y = y_coarse_arr.size

   ### NOW COMPUTE 21-21-kappa BISPECTRUM

   # Set up 2d k arrays
    kperp_arr = ell_coarse_arr/chi_A; kpar_arr = y_coarse_arr/rnu_A
    ktot_2d_arr = np.sqrt(np.outer(ell_coarse_arr,np.ones(n_coarse_y))**2/chi_A**2 + np.outer(np.ones(n_coarse_ell),y_coarse_arr)**2/rnu_A**2)
    kpar_2d_arr = np.outer(np.ones(n_coarse_ell),kpar_arr)
    kperp_2d_arr = np.outer(kperp_arr,np.ones(n_coarse_y))
    mu_k_2d_arr = kpar_2d_arr/ktot_2d_arr
    fbao_2d_k_arr = power_spec_functions.get_fbao(ktot_2d_arr)

   # GET P_m DERIVATIVE
    k_arr = kpar_arr   # USE kpar_arr to get derivative since y_arr gives smaller kmin and larger kmax
    Pm_z0_karr=power_spec_functions.get_P_m_0(k_arr)
    dPmdk_z0_arr=scipy.misc.derivative(power_spec_functions.get_P_m_0, k_arr,dx=1e-3)
    dPmdk_z0_arr_interp = scipy.interpolate.interp1d(dPmdk_z0_arr,k_arr,bounds_error=False,fill_value=0.0)
    d2Pmdk_z0_arr=scipy.misc.derivative(dPmdk_z0_arr_interp, k_arr,dx=1e-3)
    dfbao_dk = scipy.misc.derivative(power_spec_functions.get_fbao, k_arr,dx=1e-3)

   # Now evaluate all terms in bispectrum

    Pm_z0_ktot_2d_arr[:,] = np.interp(ktot_2d_arr[:,],k_arr,Pm_z0_karr)
    dPmdk_z0_ktot_2d_arr[:,]= np.interp(ktot_2d_arr[:,],k_arr,dPmdk_z0_arr) # mtx_sq[:,]=np.interp(mtx[:,],x,y)
    d2Pmdk_z0_ktot_2d_arr[:,] =np.interp(ktot_2d_arr[:,],k_arr,d2Pmdk_z0_arr)

    F_bias_rsd_2d_arr = (power_spec_functions.get_HI_bias(z_A, bias_var)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)

    ##***********************************************##
    ## *** Updated Bispec kernel *****
    ##***********************************************##

    b2 = power_spec_functions.get_HI_bias_2nd_order(z_A)
    f = power_spec_functions.get_growth_factor_f(z_A, gamma_var)
    mk = mu_k_2d_arr
    b1 = power_spec_functions.get_HI_bias(z_A, bias_var)
    Z1 = (b1 + f*mk**2.)

    Updated_derivative_term = (1/3.)*(f*mk**2. - mk**2. + 2.)*(3.0 - dPmdk_z0_ktot_2d_arr*ktot_2d_arr/Pm_z0_ktot_2d_arr)
    Updated_derivative_independent_term = (1/(14*Z1))*(14*b1*f*mk**2. + 14*b1*(f/3.) + 26.*b1*(mk**2./3.) + 26.*(b1/3.) + 28*b2 + 14*f**2.*mk**4. - 14*f**2*mk**2  \
    -6*f*mk**4. + 38*f*(mk**2/3.))

    Kernel_2d_arr_updated = Updated_derivative_term + Updated_derivative_independent_term


    Kernel_2d_arr = Kernel_2d_arr_updated


    P_21_z0_2d_arr = F_bias_rsd_2d_arr**2 *power_spec_functions.get_P_m_0(ktot_2d_arr) * power_spec_functions.get_mean_temp(z_A)**2

    volume_A_max = cd.comoving_volume(z_A_max, **cosmo); volume_A_min = cd.comoving_volume(z_A_min, **cosmo)
    volume_A = (volume_A_max - volume_A_min) *FOV_A_sr/(4.0*np.pi)  #  # - CHECK THIS


    W_kappa_over_chi_sq=lensing_kernel.get_lensing_kernel_real_sp(chi_A) / chi_A**2

    Volume_lensing_kernel_growth_factor = W_kappa_over_chi_sq * power_spec_functions.get_growth_function_D(z_A)**4 * volume_A / (chi_A**2 * rnu_A)

    C_ell_y_zA =  Integral_ell_y_zA_coarse* P_21_z0_2d_arr * Volume_lensing_kernel_growth_factor * Kernel_2d_arr
    C_ell_y_zA_fbao =  Integral_ell_y_zA_coarse_fbao* P_21_z0_2d_arr * Volume_lensing_kernel_growth_factor * Kernel_2d_arr

    f_zA = power_spec_functions.get_growth_factor_f(z_A, bias_var)
    drsd_du2 = 2.*f_zA/ F_bias_rsd_2d_arr

    dlogpk_dk = Fpm.logpk_derivative(P_spl_k, ktot_2d_arr) # Numerical deriv.
    daperp_u2 = -2. * (kperp_2d_arr/kpar_2d_arr * aperp/apar * mu_k_2d_arr**2.)**2. / aperp
    dapar_u2 =   2. * (kperp_2d_arr/kpar_2d_arr* aperp/apar * mu_k_2d_arr**2.)**2. / apar
    daperp_k = (aperp*kperp_2d_arr)**2. / (ktot_2d_arr*aperp)
    daperp_q = (aperp*q_perp_2d_arr)**2. / (q_mag_2d_arr*aperp)
    dapar_k  = (apar*kpar_2d_arr)**2. / (ktot_2d_arr*apar)
    dapar_q = (apar*q_par_2d_arr)**2. / (q_mag_2d_arr*apar)
    dKernel_term_1=4.*power_spec_functions.get_growth_factor_f(z_A, gamma_var)*(1+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)
    dKernel_term_2 = 2*power_spec_functions.get_growth_factor_f(z_A, gamma_var)**2. * (2*mu_k_2d_arr**2-1.0)
    dKernel_term_3 = power_spec_functions.get_growth_factor_f(z_A, gamma_var)*(power_spec_functions.get_HI_bias_2nd_order(z_A)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*(2*mu_k_2d_arr**2-1.0)*2*(1+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2))
    dKernel_term_4 = 0.5*power_spec_functions.get_growth_factor_f(z_A, gamma_var)*(3.0 - dPmdk_z0_ktot_2d_arr*ktot_2d_arr/Pm_z0_ktot_2d_arr)
    dkernel_2d_arr = (((dKernel_term_1 + dKernel_term_2)*F_bias_rsd_2d_arr - dKernel_term_3)/F_bias_rsd_2d_arr**2.) + dKernel_term_4
    dlogPm_dlogk = d2Pmdk_z0_ktot_2d_arr * (ktot_2d_arr/Pm_z0_ktot_2d_arr) + dPmdk_z0_ktot_2d_arr*(Pm_z0_ktot_2d_arr-ktot_2d_arr*dPmdk_z0_ktot_2d_arr)/Pm_z0_ktot_2d_arr**2.
    ### For nonlinear cases sigma_nl must add extra terms dlogpk_dk + drsd_dk + dbias_k below as in Bull et al.
    dIntegral_ell_y_zA_over_qpar_aperp = np.trapz(0.5/np.pi**2 * d_integrand *q_perp_2d_arr*daperp_q, x=qperp_arr, axis=0)
    dIntegral_ell_y_zA_over_qpar_qperp_aperp = np.trapz(dIntegral_ell_y_zA_over_qpar_aperp, x=qpar_arr, axis=0)
    dIntegral_ell_y_zA_alternative = 2.*np.pi**2. * np.amax(Pm_q_2d_arr*K_21_qperp_2d_arr*K_kappa_qperp_2d_arr*K_21_qpar_2d_arr*K_kappa_qpar_2d_arr*q_perp_2d_arr*dapar_q)

    dIntegral_ell_y_zA_over_qpar_apar= np.trapz(0.5/np.pi**2 * d_integrand *q_perp_2d_arr*dapar_q, x=qperp_arr, axis=0)
    dIntegral_ell_y_zA_over_qpar_qperp_apar = np.trapz(dIntegral_ell_y_zA_over_qpar_apar, x=qpar_arr, axis=0)
    d2Pm_dk_aperp_apar = d2Pmdk_z0_ktot_2d_arr*(ktot_2d_arr/Pm_z0_ktot_2d_arr) + dPmdk_z0_ktot_2d_arr*(Pm_z0_ktot_2d_arr-ktot_2d_arr*dPmdk_z0_ktot_2d_arr)/(Pm_z0_ktot_2d_arr**2.)

    ##***********************************************##
    ## *** Updated Derivatives for kernel *****
    ##***********************************************##

    d_derivative_term_dmu = (3.0 - dPmdk_z0_ktot_2d_arr*ktot_2d_arr/Pm_z0_ktot_2d_arr)*(2.*f*mk/3. - 2.*mk/3.)
    d_derivative_independent_term_dmu =  (1./(Z1*14.))*(-b1*(-14.*f*mk - 26.*mk/3.) + b1*(14.*f*mk + 26.*mk/3.) - f*mk*(-21.*f*mk**2. + 7.*f + 9.*mk**2. - 19./3.) + f*mk*(21.*f*mk**2. - 7.*f - 9.*mk**2. + 19./3.) - f*(-7.*f*mk**3. + 7.*f*mk + 3.*mk**3. - 19.*mk/3.) + f*(7.*f*mk**3. - 7.*f*mk - 3.*mk**3. + 19.*mk/3.))
    dkernel_2d_arr_dmu = 2*mk*(d_derivative_term_dmu  + d_derivative_independent_term_dmu )


    deriv_P21_aperp_over_P21 = ( (2./aperp) + drsd_du2 * daperp_u2 \
                        + (dlogpk_dk)*daperp_k )
    deriv_P21_apar_over_P21 =  ( (1./apar)  + drsd_du2 * dapar_u2 \
                        + (dlogpk_dk)*dapar_k  )

    deriv_aperp = ( (4./aperp) + deriv_P21_aperp_over_P21  + Soln_K_aperp\
        +(dkernel_2d_arr_dmu*daperp_u2-dlogPm_dlogk*0.5*(1+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)*daperp_k )\
			/Kernel_2d_arr ) * C_ell_y_zA
    deriv_apar =   ( (1./apar) + deriv_P21_apar_over_P21  + Soln_K_apar\
        +(dkernel_2d_arr_dmu*dapar_u2-dlogPm_dlogk *0.5*(1+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)*dapar_k )\
			/Kernel_2d_arr ) * C_ell_y_zA

    Abao_zA,sig8, b1_zA, b2_zA, Mv_zA, f_zA, aperp, apar = fid_params
    sig8_zA = sig8*power_spec_functions.get_growth_function_D(z_A)


    dB_ell_y_dAbao = C_ell_y_zA*(fbao_2d_k_arr/(1.+fbao_2d_k_arr) + Integral_ell_y_zA_coarse_fbao/Integral_ell_y_zA_coarse \
							-0.5*(1+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)\
							*(ktot_2d_arr/ (1+power_spec_functions.get_growth_factor_f(z_A, gamma_var))**2. )*dfbao_dk  )

    sig8_deriv_kernel = 4.0 * Kernel_2d_arr / (sig8_zA*power_spec_functions.get_growth_function_D(z_A)) 	# TESTING sig8 INFO
    dB_ell_y_dsig8_zA = C_ell_y_zA * sig8_deriv_kernel/Kernel_2d_arr

    dB_ell_y_db2_zA = (2/Z1)* C_ell_y_zA/Kernel_2d_arr

    dB_ell_y_db1_zA = ((14.*f*mk**2. + 14.*f/3. + 26.*mk**2./3. + 26./3.)/14./(14.*Z1))* C_ell_y_zA/Kernel_2d_arr

    f_deriv_kernel =(1/(14.*Z1))*(-b1*(-7.*mk**2. - 7./3.) + b1*(7.*mk**2. + 7./3.) - f*mk*(-7.*mk**3. + 7.*mk) + f*mk*(7.*mk**3. - 7.*mk) - mk*(-7.*f*mk**3. + 7.*f*mk + 3.*mk**3. - 19.*mk/3.) + mk*(7.*f*mk**3. - 7.*f*mk - 3.*mk**3. + 19.*mk/3.))
    dB_ell_y_df_zA =  C_ell_y_zA*(2*mu_k_2d_arr**2/F_bias_rsd_2d_arr + f_deriv_kernel/Kernel_2d_arr)


    #dB_ell_y_Mv = (logpMv_derivative(ktot_2d_arr)/P_21_z0_2d_arr + (Integral_ell_y_zA_over_qpar_qperp_Mv/Integral_ell_y_zA_coarse ) )  *C_ell_y_zA
    dB_ell_y_Mv = ((Fpm.logpMv_derivative_2(ktot_2d_arr)/power_spec_functions.get_P_m_0(ktot_2d_arr))/P_21_z0_2d_arr + (Integral_ell_y_zA_over_qpar_qperp_Mv/Integral_ell_y_zA_coarse ) )  *C_ell_y_zA


    return dB_ell_y_dAbao, dB_ell_y_dsig8_zA, dB_ell_y_db1_zA, dB_ell_y_db2_zA, dB_ell_y_Mv, dB_ell_y_df_zA, deriv_aperp, deriv_apar



def get_three_param_FisherMatrix_Cl_21_21_auto_Neutrino_mass(zbin_prop, num_k_bins_ell = 15, num_k_bins_y=14):

	## SET Z BIN PROPERTIES

    Deltanutilde = zbin_prop.Deltanutilde    # Bin width
    deltanutilde = zbin_prop.deltanutilde	 # Channel width

    z_A = zbin_prop.z_A

    nutilde_A = zbin_prop.nutilde_A
    chi_A=zbin_prop.chi_A
    rnu_A=zbin_prop.rnu_A
    lambda_A = zbin_prop.lambda_A
    FOV_A_sr = zbin_prop.FOV_A_sr
    theta_min_sr = zbin_prop.theta_min_sr

	### SET RADIAL MODES

    y_min = zbin_prop.y_min_expt ; y_max = zbin_prop.y_max_expt
    result = zbin_prop.get_y_arr_expt() ; n_y = result[0]  ; y_arr = result[1]

	### SET TRANSVERSE MODES

    ell_min = zbin_prop.ell_min_expt  ; ell_max = zbin_prop.ell_max_expt
    result = zbin_prop.get_ell_arr_expt() ; n_ell = result[0]  ; ell_arr = result[1]

    N_patches = zbin_prop.N_patches
    Mode_Volume_Factor = zbin_prop.Mode_Volume_Factor


    aperp = cosmological_parameters.aperp
    apar = cosmological_parameters.apar
    sig8 = cosmological_parameters.sig8
    Abao = cosmological_parameters.Abao
    f_nl = cosmological_parameters.f_nl
    Mv = 0.06 #eV

    ell_min_ltd = ell_min ; y_min_ltd = y_min

    non_linear_scale = zbin_prop.knl
    ell_max_ltd = non_linear_scale*chi_A
    y_max_ltd = non_linear_scale*rnu_A

    ell_arr_ltd = np.linspace(ell_min_ltd,ell_max_ltd,n_ell); y_arr_ltd = np.linspace(y_min_ltd,y_max_ltd,n_y)

    f_zA = power_spec_functions.get_growth_factor_f(z_A, gamma_var=1.0)
    b1_zA=power_spec_functions.get_HI_bias(z_A, bias_var=1.0); b2_zA = power_spec_functions.get_HI_bias_2nd_order(z_A)


    fid_fisher_params = np.array([Abao,sig8, b1_zA, b2_zA, Mv, f_zA, aperp, apar])
    n_fisher_params = fid_fisher_params.size

    Cl_Abao, Cl_sig8, Cl_b1, Cl_b2, Cl_Mv, Cl_f, Cl_aperp, Cl_apar = Cl21_Distance_derivs_Neutrino_mass(zbin_prop, ell_arr_ltd, y_arr_ltd, fid_fisher_params, bias_var=1., gamma_var=1.)

    Cl_signal = Fpm.Cl_21_auto_ell_y_zA(ell_arr_ltd, y_arr_ltd, zbin_prop, bias_var=1., gamma_var=1.)

    parameter_deriv_C_ell_y_arr = np.array([Cl_Abao,Cl_sig8, Cl_b1, Cl_b2, Cl_Mv, Cl_f, Cl_aperp, Cl_apar])


    ell_2d_arr_ltd=np.outer(ell_arr_ltd,np.ones(n_y))

    Cl_21_noise_ell_arr=Fpm.HI_angular_noise_ell_y(ell_arr_ltd, zbin_prop, experiment_name='hirax', mode='interferometer', show_plots=False)
    Cl_21_noise_ell_y_2d_arr=  np.outer(Cl_21_noise_ell_arr,np.ones(n_y))
    Cl_21_noise_ell_y_2d_arr_allsky = Fpm.HI_angular_noise_ell_y_allsky(Cl_21_noise_ell_y_2d_arr)

    Cl_21_auto_ell_y_2d_arr=Fpm.Cl_21_auto_ell_y_zA(ell_arr_ltd, y_arr_ltd, zbin_prop, bias_var=1., gamma_var=1.) # 21cm angular PS at z vs ell,y

    variance_arr = (Cl_21_noise_ell_y_2d_arr_allsky + Cl_21_auto_ell_y_2d_arr)**2


    fisher_arr_zA_21_21 = np.zeros((n_fisher_params,n_fisher_params)) ; corr_arr_zA_21_21 = np.zeros((n_fisher_params,n_fisher_params))

    for ii, pp_i in enumerate(fid_fisher_params):
		deriv_arr_ii = parameter_deriv_C_ell_y_arr[ii,]
		for jj, pp_j in enumerate(fid_fisher_params):
			deriv_arr_jj = parameter_deriv_C_ell_y_arr[jj,]
			signal_arr_sq = deriv_arr_ii * deriv_arr_jj
			fisher_ratio_arr = signal_arr_sq/variance_arr
			fisher_arr_zA_21_21[ii,jj] = 0.5 * Mode_Volume_Factor/(2*np.pi**2) * np.trapz(np.trapz(fisher_ratio_arr*ell_2d_arr_ltd, ell_arr_ltd, axis=0), y_arr_ltd, axis=0)


    fid_param_label = ['Abao','sig8','b1','b2', 'Mv','f', 'aperp', 'apar']

    return fid_param_label, fid_fisher_params, fisher_arr_zA_21_21


def get_Clkappa_distance_params_fisher_Neutrino_mass(zbin_prop, num_k_bins_ell = 15, num_k_bins_y=14):

	## SET Z BIN PROPERTIES

    Deltanutilde = zbin_prop.Deltanutilde    # Bin width
    deltanutilde = zbin_prop.deltanutilde	 # Channel width

    z_A = zbin_prop.z_A

    nutilde_A = zbin_prop.nutilde_A
    chi_A=zbin_prop.chi_A
    rnu_A=zbin_prop.rnu_A
    lambda_A = zbin_prop.lambda_A
    FOV_A_sr = zbin_prop.FOV_A_sr
    theta_min_sr = zbin_prop.theta_min_sr

    aperp = cosmological_parameters.aperp
    apar = cosmological_parameters.apar
    sig8 = cosmological_parameters.sig8
    Abao = cosmological_parameters.Abao


	### SET RADIAL MODES

    y_min = zbin_prop.y_min_expt ; y_max = zbin_prop.y_max_expt
    result = zbin_prop.get_y_arr_expt() ; n_y = result[0]  ; y_arr = result[1]

	### SET TRANSVERSE MODES

    ell_min = zbin_prop.ell_min_expt  ; ell_max = zbin_prop.ell_max_expt
    result = zbin_prop.get_ell_arr_expt() ; n_ell = result[0]  ; ell_arr = result[1]

    N_patches = zbin_prop.N_patches
    Mode_Volume_Factor = zbin_prop.Mode_Volume_Factor


    sig8 = cosmological_parameters.sig8
    h = cosmological_parameters.h
    Om_m = cosmological_parameters.om_m
    Om_DE = cosmological_parameters.om_L
    w0 = cosmological_parameters.w0
    wa = cosmological_parameters.wa
    Mv = 0.06 #eV


    ell_min_ltd = ell_min ; y_min_ltd = y_min
    non_linear_scale = zbin_prop.knl
    ell_max_ltd = non_linear_scale*chi_A
    y_max_ltd = non_linear_scale*rnu_A
    ell_arr_ltd = np.linspace(ell_min_ltd,ell_max_ltd,n_ell); y_arr_ltd = np.linspace(y_min_ltd,y_max_ltd,n_y)

    Omega_HI_zA=power_spec_functions.get_Omega_HI(z_A) ; f_zA = power_spec_functions.get_growth_factor_f(z_A, gamma_var=1.0)
    b1_zA=power_spec_functions.get_HI_bias(z_A, bias_var=1.0); b2_zA = power_spec_functions.get_HI_bias_2nd_order(z_A)

    fid_fisher_params = np.array([sig8, Mv, Om_DE, w0, wa, h])
    n_fisher_params = fid_fisher_params.size

    Cl_sig8, Cl_Mv, Cl_Om_DE, Cl_w0, Cl_wa, Cl_h = Cl_kappa_distance_derivs_Neutrino_mass(ell_arr_ltd, y_arr_ltd, zbin_prop, fid_fisher_params, bias_var=1., gamma_var=1.)

    Cl_signal = Fpm.CMB_convergence_power(ell_arr_ltd)

    parameter_deriv_C_ell_y_arr = np.array([Cl_sig8, Cl_Mv, Cl_Om_DE, Cl_w0, Cl_wa, Cl_h])  #,Cl_Omega_HI

    ell_2d_arr_ltd=np.outer(ell_arr_ltd,np.ones(n_y))

    Cl_kappa_noise_ell_arr=Fpm.CMB_convergence_noise(ell_arr_ltd)
    Cl_kappa_noise_ell_y_2d_arr=  np.outer(Cl_kappa_noise_ell_arr,np.ones(n_y))
    Cl_kappa_auto_ell_y_2d_arr= np.outer(Cl_signal,np.ones(n_y))

    variance_arr = (Cl_kappa_noise_ell_y_2d_arr+ Cl_kappa_auto_ell_y_2d_arr)**2.

    fisher_arr_zA_kappa = np.zeros((n_fisher_params,n_fisher_params)) ; corr_arr_zA_kappa = np.zeros((n_fisher_params,n_fisher_params))

    for ii, pp_i in enumerate(fid_fisher_params):
		deriv_arr_ii = parameter_deriv_C_ell_y_arr[ii,]
		for jj, pp_j in enumerate(fid_fisher_params):
			deriv_arr_jj = parameter_deriv_C_ell_y_arr[jj,]
			signal_arr_sq = deriv_arr_ii * deriv_arr_jj
			fisher_ratio_arr = signal_arr_sq/variance_arr
			fisher_arr_zA_kappa[ii,jj] = 0.5 * Mode_Volume_Factor/(2.*np.pi**2) * np.trapz(np.trapz(fisher_ratio_arr*ell_2d_arr_ltd, ell_arr_ltd, axis=0), y_arr_ltd, axis=0)

    fid_param_label = ['sig8', 'Mv', 'Om_DE', 'w0', 'wa', 'h'] #, 'Omega_HI'

    return fid_param_label, fid_fisher_params, fisher_arr_zA_kappa


def get_distance_params_bispec_fisher_Neutrino_mass(zbin_prop, num_k_bins_ell = 15, num_k_bins_y=14):

	## SET Z BIN PROPERTIES

    Deltanutilde = zbin_prop.Deltanutilde    # Bin width
    deltanutilde = zbin_prop.deltanutilde	 # Channel width

    z_A = zbin_prop.z_A

    nutilde_A = zbin_prop.nutilde_A
    chi_A=zbin_prop.chi_A
    rnu_A=zbin_prop.rnu_A
    lambda_A = zbin_prop.lambda_A
    FOV_A_sr = zbin_prop.FOV_A_sr
    theta_min_sr = zbin_prop.theta_min_sr

    aperp = cosmological_parameters.aperp
    apar = cosmological_parameters.apar
    sig8 = cosmological_parameters.sig8
    Abao = cosmological_parameters.Abao
    fnl = cosmological_parameters.f_nl
    Mv = 0.06 #eV


	### SET RADIAL MODES

    y_min = zbin_prop.y_min_expt ; y_max = zbin_prop.y_max_expt
    result = zbin_prop.get_y_arr_expt() ; n_y = result[0]  ; y_arr = result[1]

	### SET TRANSVERSE MODES

    ell_min = zbin_prop.ell_min_expt  ; ell_max = zbin_prop.ell_max_expt
    result = zbin_prop.get_ell_arr_expt() ; n_ell = result[0]  ; ell_arr = result[1]

    N_patches = zbin_prop.N_patches
    Mode_Volume_Factor = zbin_prop.Mode_Volume_Factor

	### GET 21-21 Auto Power Spectrum - signal and noise
    Cl_21_auto_ell_y_2d_arr=Fpm.Cl_21_auto_ell_y_zA(ell_arr, y_arr, zbin_prop, bias_var=1., gamma_var=1.) # 21cm angular PS at z vs ell,y
    Cl_21_noise_ell_arr=Fpm.HI_angular_noise_ell_y(ell_arr, zbin_prop, experiment_name='hirax', mode='interferometer', show_plots=False)
    Cl_21_noise_ell_y_2d_arr=  np.outer(Cl_21_noise_ell_arr,np.ones(n_y))

    CMB_kappa_average_variance_sky = Fpm.compute_average_kappa_variance_sky_zA(zbin_prop, bias_var=1., gamma_var=1.)


    ell_min_ltd = ell_min ; y_min_ltd = y_min
    non_linear_scale = zbin_prop.knl
    ell_max_ltd = non_linear_scale*chi_A
    y_max_ltd = non_linear_scale*rnu_A
    ell_arr_ltd = np.linspace(ell_min_ltd,ell_max_ltd,n_ell); y_arr_ltd = np.linspace(y_min_ltd,y_max_ltd,n_y)

    Omega_HI_zA=power_spec_functions.get_Omega_HI(z_A) ; f_zA = power_spec_functions.get_growth_factor_f(z_A, gamma_var=1.0)
    b1_zA=power_spec_functions.get_HI_bias(z_A, bias_var=1.0); b2_zA = power_spec_functions.get_HI_bias_2nd_order(z_A)

    fid_fisher_params = np.array([Abao, sig8, b1_zA, b2_zA, Mv, f_zA, aperp, apar])
    n_fisher_params = fid_fisher_params.size

    print "STARTING FISHER MATRIX"

    Bl_Abao, Bl_sig8, Bl_b1, Bl_b2, Bl_Mv, Bl_f, Bl_aperp, Bl_apar =  bispec_distance_derivs_Neutrino_mass(ell_arr_ltd, y_arr_ltd, zbin_prop, fid_fisher_params, bias_var=1., gamma_var=1.)

    Bl_signal = Fpm.integ_bispec_kappa_21_21_ell_y_zA(ell_arr_ltd, y_arr_ltd, zbin_prop, bias_var=1., gamma_var=1.)


    parameter_deriv_ell_y_arr = np.array([Bl_Abao, Bl_sig8, Bl_b1, Bl_b2, Bl_Mv, Bl_f, Bl_aperp, Bl_apar])#,Bl_Omega_HI # parameter_deriv_ell_y_arr[ii,]

    ell_2d_arr_ltd=np.outer(ell_arr_ltd,np.ones(n_y))
    CMB_kappa_noise_ell = Fpm.CMB_convergence_noise(ell_2d_arr_ltd)
    CMB_kappa_signal_ell = Fpm.CMB_convergence_power(ell_2d_arr_ltd)

    Cl_21_noise_ell_arr=Fpm.HI_angular_noise_ell_y(ell_arr_ltd, zbin_prop, experiment_name='hirax', mode='interferometer', show_plots=False)
    Cl_21_noise_ell_y_2d_arr=  np.outer(Cl_21_noise_ell_arr,np.ones(n_y))
    Cl_21_noise_ell_y_2d_arr_allsky = Cl_21_noise_ell_y_2d_arr/fsky

    variance_arr = 6.* N_patches**2. * Bl_signal**2. + \
    (3.*fsky**2.)*(Cl_21_noise_ell_y_2d_arr_allsky + Cl_21_auto_ell_y_2d_arr)**2. * CMB_kappa_average_variance_sky # (CMB_kappa_noise_ell+CMB_kappa_signal_ell)
    fisher_arr_zA_21_21_kappa = np.zeros((n_fisher_params,n_fisher_params)) ; corr_arr_zA=np.zeros((n_fisher_params,n_fisher_params))
    variance_arr = variance_arr


    for ii, pp_i in enumerate(fid_fisher_params):
		deriv_arr_ii = parameter_deriv_ell_y_arr[ii,]
		for jj, pp_j in enumerate(fid_fisher_params):
			deriv_arr_jj = parameter_deriv_ell_y_arr[jj,]
			signal_arr_sq = deriv_arr_ii * deriv_arr_jj
			fisher_ratio_arr = N_patches**2 * signal_arr_sq/variance_arr
			fisher_arr_zA_21_21_kappa[ii,jj] = 0.5 * Mode_Volume_Factor/(2*np.pi**2) * np.trapz(np.trapz(fisher_ratio_arr*ell_2d_arr_ltd, ell_arr_ltd, axis=0), y_arr_ltd, axis=0)


    fid_param_label = ['Abao','sig8', 'b1', 'b2', 'Mv', 'f', 'aperp', 'apar'] #,'Omega_HI


    return fid_param_label, fid_fisher_params, fisher_arr_zA_21_21_kappa
