####################################################################################################################
# This module produces all the primary functions required for the Fisher analysis and signal-to-noise plots. Here we compute
# the individual auto-correlation spectra for HI and CMB lensing fields as well as the 2pt cross-correlation and 3pt integrated bispectrum.
# We also define useful helper functions here for the matter power spec derivative and neutrino mass derivative. We also define
# here the ellipse plotting functions used to plot the Fisher matrix results.
# Running this script as main will produce the 2d-bispectrum SNR plot. If the 'main' part of the code is edited one can
# choose the option to run the HI auto 2d SNR or the 2pt HI-CMB lensing cross-correlation 2d SNR.
# The binning set up and experiment are chosen/set up in the binning.py and settings.py files respectively.
################################################################################################################################


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
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


##################################################################
######## Ellipse Plotting Functions ##############
##################################################################


def plot_ellipse(semimaj=1,semimin=1,phi=0,x_cent=0,y_cent=0,theta_num=1e3,ax=None,plot_kwargs=None,\
                    fill=False,fill_kwargs=None,data_out=False,cov=None,mass_level=0.68):
    '''
        An easy to use function for plotting ellipses in Python 2.7!

        The function creates a 2D ellipse in polar coordinates then transforms to cartesian coordinates.
        It can take a covariance matrix and plot contours from it.

        semimaj : float
            length of semimajor axis (always taken to be some phi (-90<phi<90 deg) from positive x-axis!)

        semimin : float
            length of semiminor axis

        phi : float
            angle in radians of semimajor axis above positive x axis

        x_cent : float
            X coordinate center

        y_cent : float
            Y coordinate center

        theta_num : int
            Number of points to sample along ellipse from 0-2pi

        ax : matplotlib axis property
            A pre-created matplotlib axis

        plot_kwargs : dictionary
            matplotlib.plot() keyword arguments

        fill : bool
            A flag to fill the inside of the ellipse

        fill_kwargs : dictionary
            Keyword arguments for matplotlib.fill()

        data_out : bool
            A flag to return the ellipse samples without plotting

        cov : ndarray of shape (2,2)
            A 2x2 covariance matrix, if given this will overwrite semimaj, semimin and phi

        mass_level : float
            if supplied cov, mass_level is the contour defining fractional probability mass enclosed
            for example: mass_level = 0.68 is the standard 68% mass

    '''

    # Get Ellipse Properties from cov matrix
    if cov is not None:
        eig_vec,eig_val,u = np.linalg.svd(cov)
        # Make sure 0th eigenvector has positive x-coordinate
        if eig_vec[0][0] < 0:
            eig_vec[0] *= -1
        semimaj = np.sqrt(eig_val[0])
        semimin = np.sqrt(eig_val[1])
        if mass_level is None:
            multiplier = np.sqrt(2.279)
        else:
            distances = np.linspace(0,20,20001)
            chi2_cdf = chi2.cdf(distances,df=2)
            multiplier = np.sqrt(distances[np.where(np.abs(chi2_cdf-mass_level)==np.abs(chi2_cdf-mass_level).min())[0][0]])
        semimaj *= multiplier
        semimin *= multiplier
        phi = np.arccos(np.dot(eig_vec[0],np.array([1,0])))
        if eig_vec[0][1] < 0 and phi > 0:
            phi *= -1

    # Generate data for ellipse structure
    theta = np.linspace(0,2*np.pi,theta_num)
    r = 1 / np.sqrt((np.cos(theta))**2 + (np.sin(theta))**2)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    data = np.array([x,y])
    S = np.array([[semimaj,0],[0,semimin]])
    R = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
    T = np.dot(R,S)
    data = np.dot(T,data)
    data[0] += x_cent
    data[1] += y_cent

    # Output data?
    if data_out == True:
        return data

    # Plot!
    return_fig = False
    if ax is None:
        return_fig = True
        fig,ax = plt.subplots(figsize=(100,100))
        ax.locator_params(axis='x',tight=True,nbins=1)
        ax.tick_params(axis='both', which='minor', labelsize=13)
        #ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useOffset=False)

    if plot_kwargs is None:
        ax.plot(data[0],data[1],color='b',linestyle='-')
        ax.locator_params(axis='x',tight=True,nbins=1)
        ax.tick_params(axis='both', which='minor', labelsize=13)
        #ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useOffset=False)
    else:
        ax.plot(data[0],data[1],**plot_kwargs)
        ax.locator_params(axis='x',tight=True,nbins=1)
        ax.tick_params(axis='both', which='minor', labelsize=13)
        #ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useOffset=False)

    if fill == True:
        ax.fill(data[0],data[1],**fill_kwargs)
        ax.locator_params(axis='x',tight=True,nbins=1)
        ax.tick_params(axis='both', which='minor', labelsize=13)
        #ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useOffset=False)

    if return_fig == True:
        return fig
    ax.locator_params(axis='x',tight=True,nbins=1)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    #ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useOffset=False)


def plot_2d_contour(twod_mean, twod_cov_matrix):

	nsig1,nsig2 = np.sqrt(np.array([6.17, 2.3]))

	x0,y0 = twod_mean
	sigxsq, sigysq, sigxy= twod_cov_matrix[0,0], twod_cov_matrix[1,1], twod_cov_matrix[0,1]
	asq, bsq = 0.5*(sigxsq + sigysq) + 0.25 * np.sqrt((sigxsq - sigysq)**2 + sigxy**2), 0.5*(sigxsq + sigysq) - 0.25 * np.sqrt((sigxsq - sigysq)**2 + sigxy**2)

	sin2ang = 2 * sigxy ; cos2ang = sigxsq - sigysq
	ang = np.arctan(cos2ang/sin2ang) / 2.0

	print 	sin2ang, cos2ang
	#raise KeyboardInterrupt

	Ellipse = matplotlib.patches.Ellipse
	patch = Ellipse(xy=(x0,y0), width=2*nsig1*np.sqrt(asq), height=2*nsig1*np.sqrt(bsq), angle=ang*180/np.pi, fill=True)
	print 2*nsig1*np.sqrt(asq), 2*nsig1*np.sqrt(bsq), ang*180/np.pi
	patch.set_fc('b')
	patch.set_ec('b')
	patch.set_alpha(1)
	patch.set_zorder(1)
	patch.set_lw(1)
	fig = plt.figure(0)
	ax = fig.add_subplot(111)
	patch.set_lw(2)
	patch.set_clip_box(ax.bbox)
	patch.set_alpha(1)
	patch.set_facecolor([0.1,0.2,0.3])
	ax.add_artist(patch)


	plt.show()


####################################################################################
############ Log derivative of the Power Spectrum
####################################################################################


def logpk_derivative(pk, kgrid):
    """
    Calculate the first derivative of the (log) power spectrum,
    d log P(k) / d k. Sets the derivative to zero wherever P(k) is not defined.

    Parameters
    ----------

    pk : function
        Callable function (usually an interpolation fn.) for P(k)

    kgrid : array_like
        Array of k values on which the integral will be computed.
    """

    dk = 1e-7
    np.seterr(invalid='ignore')
    dP = pk(kgrid + 0.5*dk) / pk(kgrid - 0.5*dk)
    np.seterr(invalid=None)
    dP[np.where(np.isnan(dP))] = 1. # Set NaN values to 1 (sets deriv. to zero)
    dlogpk_dk = np.log(dP) / dk
    return dlogpk_dk



def logpMv_derivative_2(k_in):
    kh,P = np.loadtxt('dPm_dnu.dat') #     dPm_dnu_Omega_m_029.dat
	# Switch between fiducial mnu value and altered 0.29 version
    k_arr = kh/cosmological_parameters.h
    dlog_interp = interp1d(k_arr, P*cosmological_parameters.h**3,bounds_error=False,fill_value=0.0)
    return dlog_interp(k_in)


#################################################################################
############### POWER SPECTRA FUNCTIONS
#################################################################################



def CMB_convergence_power(ells, show_plots=False):

    ''' Load in CMB lensing PS from CAMB '''

    c_kappa_CAMB=CMB_lensing.getCkappa(ells)
    if show_plots:
        plt.loglog(ells, c_kappa_CAMB)
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'C_\ell^{\kappa}')
        plt.title('Lensing angular power')
        plt.show()
    return c_kappa_CAMB


def CMB_convergence_noise(ells, experiment_name='act', est='eb', show_plots=False):

	''' Load in CMB noise '''

	n_kappa=CMB_lensing.getN0kappa(ells, experiment_name, est)
	return n_kappa


def compute_average_kappa_variance_sky_zA(zbin_prop, bias_var=1., gamma_var=1.):

	''' Compute average variance of the convergence field in a volume '''

	z_A = zbin_prop.z_A
	Deltanutilde = zbin_prop.Deltanutilde
	deltanutilde = zbin_prop.deltanutilde
	nutilde_A = zbin_prop.nutilde_A
	chi_A = zbin_prop.chi_A
	rnu_A = zbin_prop.rnu_A
	lambda_A = zbin_prop.lambda_A
	nu_tilde_min = zbin_prop.nu_tilde_min; nu_tilde_max = zbin_prop.nu_tilde_max;
	z_A_min = zbin_prop.z_A_min ; z_A_max = zbin_prop.z_A_max;
	chi_A_min = zbin_prop.chi_A_min; chi_A_max = zbin_prop.chi_A_max;
	delta_chipar_bin = zbin_prop.delta_chipar_bin
	FOV_A_sr = zbin_prop.FOV_A_sr
	theta_min_sr = zbin_prop.theta_min_sr
	delta_chiperp_bin = zbin_prop.delta_chiperp_bin

	### DO INTEGRAL OF MATTER POWER SPECTRUM OVER qpar, qperp

	qpar_min = zbin_prop.qpar_min ; qpar_max = zbin_prop.qpar_max

	qperp_min = zbin_prop.qperp_min ; qperp_max = zbin_prop.qperp_max

	num_qpar = 1000 ; num_qperp = 999
	qpar_arr = np.linspace(qpar_min,qpar_max,num_qpar) ; qperp_arr = np.linspace(qperp_min,qperp_max,num_qperp)
	q_perp_2d_arr = np.outer(qperp_arr,np.ones(num_qpar))
	q_par_2d_arr = np.outer(np.ones(num_qperp),qpar_arr)

	K_kappa_qpar_arr = np.sin(qpar_arr*delta_chipar_bin/2.0)/ (qpar_arr*delta_chipar_bin/2.0)
	K_kappa_qperp_arr = np.sin(qperp_arr*delta_chiperp_bin/2.0)/(qperp_arr*delta_chiperp_bin/2.0)
	K_kappa_qperp_2d_arr = np.outer(K_kappa_qperp_arr,np.ones(num_qpar))  ; K_kappa_qpar_2d_arr = np.outer(np.ones(num_qperp), K_kappa_qpar_arr)

	K_kappa_2d_arr= K_kappa_qperp_2d_arr*K_kappa_qpar_2d_arr

	ell_arr = qperp_arr * chi_A

	W_kappa_over_chi_sq=lensing_kernel.get_lensing_kernel_real_sp(chi_A)**2. / chi_A**2.
	variance_kappa_qperp_arr = CMB_convergence_noise(ell_arr) + CMB_convergence_power(ell_arr)

	variance_kappa_qperp_qpar_arr = np.outer(variance_kappa_qperp_arr,np.ones(num_qpar))

	Integral_kappa_var_zA_over_qpar = np.trapz(0.5/np.pi**2 * variance_kappa_qperp_qpar_arr*K_kappa_2d_arr**2*q_perp_2d_arr, x=qperp_arr, axis=0)
	Integral_kappa_var_zA_over_qpar_qperp = np.trapz(Integral_kappa_var_zA_over_qpar, x=qpar_arr, axis=0)

	Integral_kappa_var_zA_over_qpar2 = np.trapz(0.5/np.pi**2 * variance_kappa_qperp_qpar_arr*K_kappa_2d_arr**2*q_perp_2d_arr, x=qperp_arr, axis=0)
	Integral_kappa_var_zA_over_qpar_qperp2 = np.trapz(Integral_kappa_var_zA_over_qpar, x=qpar_arr, axis=0)

	volume_A_max = cd.comoving_volume(z_A_max, **cosmo); volume_A_min = cd.comoving_volume(z_A_min, **cosmo)
	volume_A = (volume_A_max - volume_A_min) *FOV_A_sr/(4.0*np.pi)

	CMB_kappa_average_variance_sky = volume_A * Integral_kappa_var_zA_over_qpar_qperp

	return CMB_kappa_average_variance_sky



def HI_angular_noise_ell_y(ells, zbin_prop,T_obs = 2., experiment_name='hirax', mode='interferometer', show_plots=False):

   ''' Compute the HI noise power spectrum for a specific experiment from the HI_experiments code '''

   z_A = zbin_prop.z_A
   nu=1420e6/(z_A+1.)
   expt=HI_experiments.getHIExptObject(experiment_name, mode, tobsyears = T_obs)
   noise_HI_ell_y=expt.getNoiseAngularPower(ells, nu,0.4e6)

   return noise_HI_ell_y


def HI_angular_noise_ell_y_allsky(HI_angular_noise_ell_y):

    return HI_angular_noise_ell_y/fsky


def Cl_21_auto_ell_y_zA(ell_arr, y_arr, zbin_prop, bias_var=1., gamma_var=1.):

   ''' Compute the 2d HI PS in a given redshift bin '''

   n_ell = ell_arr.size ; n_y = y_arr.size
   z_A = zbin_prop.z_A
   chi_A = zbin_prop.chi_A ; nutilde_A = zbin_prop.nutilde_A ; rnu_A = zbin_prop.rnu_A
   kperp_arr = ell_arr/chi_A; kpar_arr = y_arr/rnu_A

   ktot_2d_arr = np.sqrt(np.outer(ell_arr,np.ones(n_y))**2/chi_A**2 + np.outer(np.ones(n_ell),y_arr)**2/rnu_A**2)
   kpar_2d_arr = np.outer(np.ones(n_ell),kpar_arr)
   kperp_2d_arr = np.outer(kperp_arr,np.ones(n_y))
   mu_k_2d_arr = kpar_2d_arr/ktot_2d_arr

   F_bias_rsd_sq_2d_arr = (power_spec_functions.get_HI_bias(z_A, bias_var)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)**2
   P_21_tot_2d_arr = F_bias_rsd_sq_2d_arr *power_spec_functions.get_P_m_0(ktot_2d_arr) * \
		power_spec_functions.get_mean_temp(z_A)**2 * power_spec_functions.get_growth_function_D(z_A)**2  # mK^2

   Cl_21_auto_ell_y_2d_arr = P_21_tot_2d_arr / (chi_A**2 * rnu_A)

   return Cl_21_auto_ell_y_2d_arr



def Cl_21_kappa_cross_ell_y(ell_arr, y_arr, zbin_prop, bias_var=1., gamma_var=1.):

   ''' Compute the HI-CMB lensing 2-point cross-correlation PS '''

   n_ell = ell_arr.size ; n_y = y_arr.size
   z_A = zbin_prop.z_A
   chi_A = zbin_prop.chi_A ; nutilde_A = zbin_prop.nutilde_A ; rnu_A = zbin_prop.rnu_A
   kperp_arr = ell_arr/chi_A; kpar_arr = y_arr/rnu_A

   ktot_2d_arr = np.sqrt(np.outer(ell_arr,np.ones(n_y))**2/chi_A**2 + np.outer(np.ones(n_ell),y_arr)**2/rnu_A**2)
   kpar_2d_arr = np.outer(np.ones(n_ell),kpar_arr)
   kperp_2d_arr = np.outer(kperp_arr,np.ones(n_y))
   mu_k_2d_arr = kpar_2d_arr/ktot_2d_arr

   lensing_kernel_arr = lensing_kernel.get_lensing_kernel_fourier_sp(kpar_arr)
   lensing_kernel_2d_arr = np.outer(np.ones(n_ell),lensing_kernel_arr)

   F_bias_rsd_2d_arr = (power_spec_functions.get_HI_bias(z_A, bias_var)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)
   Cl_21_kappa_cross_ell_y_2d_arr = power_spec_functions.get_mean_temp(z_A) * F_bias_rsd_2d_arr * power_spec_functions.get_growth_function_D(z_A) * \
		lensing_kernel_2d_arr * power_spec_functions.get_P_m_0(ktot_2d_arr)/ (chi_A**2 * rnu_A) # mK

   return Cl_21_kappa_cross_ell_y_2d_arr



def integ_bispec_kappa_21_21_ell_y_zA(ell_coarse_arr, y_coarse_arr, zbin_prop, bias_var=1., gamma_var=1.):

   ''' Compute the HI-HI-CMB lensing integrated bispectrum '''

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

	### DO INTEGRAL OF MATTER POWER SPECTRUM OVER qpar, qperp

   qpar_min = zbin_prop.qpar_min ; qperp_min = zbin_prop.qperp_min
   num_qpar = 500 ; num_qperp = 499
   N_factor_integral_range=200
   qpar_arr = np.linspace(1e-7,N_factor_integral_range*qpar_min,num_qpar) ; qperp_arr = np.linspace(1e-7,N_factor_integral_range*qperp_min,num_qperp)
   q_mag_min = np.sqrt( qpar_min**2. + qperp_min**2.)
   q_mag_max = np.sqrt( qpar_arr[-1]**2. + qperp_arr[-1]**2. )

   K_21_qpar_arr = np.sin(qpar_arr*delta_chipar_bin/2.0)/ (qpar_arr*delta_chipar_bin/2.0)
   K_21_qperp_arr = np.sin(qperp_arr*delta_chiperp_bin/2.0)/(qperp_arr*delta_chiperp_bin/2.0)
   K_21_qpar_arr[0] = 1.0
   K_21_qperp_arr[0] = 1.0
   K_kappa_qperp_arr = K_21_qperp_arr
   K_kappa_qpar_arr = K_21_qpar_arr

   K_21_qperp_2d_arr = np.outer(K_21_qperp_arr,np.ones(num_qpar))  ; K_21_qpar_2d_arr = np.outer(np.ones(num_qperp), K_21_qpar_arr)
   K_kappa_qperp_2d_arr = np.outer(K_kappa_qperp_arr,np.ones(num_qpar))  ; K_kappa_qpar_2d_arr = np.outer(np.ones(num_qperp), K_kappa_qpar_arr)

   q_par_2d_arr = np.outer(np.ones(num_qperp),qpar_arr)
   q_perp_2d_arr = np.outer(qperp_arr,np.ones(num_qpar))

   q_mag_2d_arr = np.sqrt(np.outer(qperp_arr,np.ones(num_qpar))**2 + np.outer(np.ones(num_qperp),qpar_arr)**2)

   Pm_q_2d_arr = power_spec_functions.get_P_m_0(q_mag_2d_arr)
   dPmdq_z0_ktot_2d_arr = np.zeros((qperp_arr.size,qpar_arr.size))

   dPmdq_z0_arr=scipy.misc.derivative( power_spec_functions.get_P_m_0, np.sqrt(qperp_arr**2.),dx=1e-3)

   dPmdq_z0_ktot_2d_arr[:,]= np.interp(q_mag_2d_arr[:,],qperp_arr,dPmdq_z0_arr)

   Integral_ell_y_zA_over_qpar = np.trapz(0.5/np.pi**2 * Pm_q_2d_arr*K_21_qperp_2d_arr*K_kappa_qperp_2d_arr*K_21_qpar_2d_arr*K_kappa_qpar_2d_arr*q_perp_2d_arr, x=qperp_arr, axis=0)
   Integral_ell_y_zA_over_qpar_qperp = np.trapz(Integral_ell_y_zA_over_qpar, x=qpar_arr, axis=0)

   PT_Integral_ell_y_zA_over_qpar = np.trapz(0.5/np.pi**2 *K_21_qperp_2d_arr*K_kappa_qperp_2d_arr*K_21_qpar_2d_arr*K_kappa_qpar_2d_arr*q_perp_2d_arr*dPmdq_z0_ktot_2d_arr, x=qperp_arr, axis=0)
   PT_Integral_ell_y_zA_over_qpar_qperp = np.trapz(PT_Integral_ell_y_zA_over_qpar, x=qpar_arr, axis=0)


   Integral_ell_y_zA_coarse = Integral_ell_y_zA_over_qpar_qperp
   PT_Integral_ell_y_zA_coarse = PT_Integral_ell_y_zA_over_qpar_qperp

   C_ell_y_zA = np.zeros((ell_coarse_arr.size,y_coarse_arr.size))
   dPmdk_z0_ktot_2d_arr = np.zeros((ell_coarse_arr.size,y_coarse_arr.size))
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

	# GET P_m DERIVATIVE
   k_arr = kpar_arr   # USE kpar_arr to get derivative since y_arr gives smaller kmin and larger kmax
   Pm_z0_karr=power_spec_functions.get_P_m_0(k_arr)
   dPmdk_z0_arr=scipy.misc.derivative(power_spec_functions.get_P_m_0, k_arr,dx=1e-3)

	# Now evaluate all terms in bispectrum

   Pm_z0_ktot_2d_arr[:,] = np.interp(ktot_2d_arr[:,],k_arr,Pm_z0_karr)
   dPmdk_z0_ktot_2d_arr[:,]= np.interp(ktot_2d_arr[:,],k_arr,dPmdk_z0_arr)

   F_bias_rsd_2d_arr = (power_spec_functions.get_HI_bias(z_A, bias_var)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)

   ##***********************************************##
   ## *** Bispectrum kernel *****
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

   P_21_z0_2d_arr = F_bias_rsd_2d_arr**2 *power_spec_functions.get_P_m_0(ktot_2d_arr) * power_spec_functions.get_mean_temp(z_A)**2

   volume_A_max = cd.comoving_volume(z_A_max, **cosmo); volume_A_min = cd.comoving_volume(z_A_min, **cosmo)
   volume_A = (volume_A_max - volume_A_min) *FOV_A_sr/(4.0*np.pi)

   W_kappa_over_chi_sq=lensing_kernel.get_lensing_kernel_real_sp(chi_A) / chi_A**2.

   Volume_lensing_kernel_growth_factor = W_kappa_over_chi_sq * power_spec_functions.get_growth_function_D(z_A)**4 * volume_A / (chi_A**2 * rnu_A)

   bispec_amp = Volume_lensing_kernel_growth_factor*power_spec_functions.get_mean_temp(z_A)*power_spec_functions.get_HI_bias(z_A, bias_var)\
   					*300/chi_A *(( power_spec_functions.get_HI_bias(z_A, bias_var) - 2*power_spec_functions.get_growth_factor_f(z_A, gamma_var)/power_spec_functions.get_HI_bias(z_A, bias_var) ) +1 ) \
					*(q_mag_max**2 -  q_mag_min**2)/2.

   C_ell_y_zA_updated =  Integral_ell_y_zA_coarse* P_21_z0_2d_arr * Volume_lensing_kernel_growth_factor *Kernel_2d_arr_updated


   return C_ell_y_zA_updated



def compute_SNR_Cl_21_kappa_cross(zbin_prop, k_min = 1.e-7, k_max = 3.0e0,
                                  n_ell_full = 1000, n_y_full = 5000, num_k_bins_ell = 15,
                                  num_k_bins_y=14, delta_kpar = 0.01, delta_kperp = 0.01, kcut=0.001):


    ''' Compute the SNR and plot for the HI-CMB lensing 2pt cross-correlation '''

	## SET Z BIN PROPERTIES

    Deltanutilde = zbin_prop.Deltanutilde    # Bin width

    Mode_Volume_Factor = zbin_prop.Mode_Volume_Factor


    nutilde_A = zbin_prop.nutilde_A
    chi_A=zbin_prop.chi_A; z_A=zbin_prop.z_A
    rnu_A=zbin_prop.rnu_A
    FOV_A_sr = zbin_prop.FOV_A_sr
    chi_max = zbin_prop.chi_A_min; chi_min = zbin_prop.chi_A_max
	### SET RADIAL MODES
    y_min = zbin_prop.y_min_expt ; y_max = zbin_prop.y_max_expt
    result = zbin_prop.get_y_arr_expt() ; n_y = result[0]  ; y_arr = result[1]

	### SET TRANSVERSE MODES
    ell_min = zbin_prop.ell_min_expt  ; ell_max = zbin_prop.ell_max_expt
    result = zbin_prop.get_ell_arr_expt() ; n_ell = result[0]  ; ell_arr = result[1]   # CHANGE THESE TO _expt
    N_patches = zbin_prop.N_patches
    Mode_Volume_Factor = zbin_prop.Mode_Volume_Factor

	### COMPUTE CROSS 21-kappa POWER SPECTRUM AND SIGNAL-TO-NOISE

    y_min_full = k_min * rnu_A ; y_max_full = k_max * rnu_A

    ell_max_full = ell_max ; ell_min_full = 20
    y_arr_full=np.logspace(np.log10(y_min_full),np.log10(y_max_full),n_y_full)
    ell_arr_full=np.logspace(np.log10(ell_min_full),np.log10(ell_max_full),n_ell_full)

    Kernel_arr0_1 = np.loadtxt('Full_HIRAX_Lensing_Kernel_kpar.dat') #lensing_kernel.get_lensing_kernel_fourier_sp_bin(y_arr_full/rnu_A,2798., 6032.)

    Cl_21_kappa_cross_ell_y_2d_arr= Cl_21_kappa_cross_ell_y(ell_arr_full, y_arr_full, zbin_prop, bias_var=1., gamma_var=1.)
    Cl_21_noise_ell_arr=HI_angular_noise_ell_y(ell_arr_full, zbin_prop, experiment_name='hirax', mode='interferometer', show_plots=False)
    CMB_kappa_noise_ell = CMB_convergence_noise(ell_arr_full)

    y_plot = np.linspace(1,y_arr_full[-1],y_arr_full[-1]+1)
    Cros_plot = Cl_21_kappa_cross_ell_y(ell_arr_full, y_plot, zbin_prop, bias_var=1., gamma_var=1.)

    plt.loglog(ell_arr_full,np.sqrt(HI_angular_noise_ell_y_allsky(Cl_21_noise_ell_arr))*np.sqrt(CMB_kappa_noise_ell)/np.sqrt(fsky * ell_arr_full),'-.')
    Clcross_plot=np.abs(Cl_21_kappa_cross_ell_y_2d_arr[:,0:n_y_full:1000])
    for i in np.arange(len(y_arr_full[0:n_y_full:1000])):
        y=y_arr_full[100:n_y_full:800]
        plt.plot(ell_arr_full, Clcross_plot[:,i],label='y='+str(round(y[i],2)))
    plt.legend(loc='lower right',fontsize=13)
    plt.xlabel('$\ell$',fontsize=15); plt.ylabel('$C^{21\,\kappa}_\ell(y) \quad vs \quad \sqrt{N^{21}_\ell(y) N^{cmb}_\ell/(f_{sky} \,\ell)} \,\, (mK)$',fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.ylim(1e-17,1e-7)
    if SAVEFIG: plt.savefig(output_dir+'figure_HI_kappa_signal_noise_1d.png')
    if SHOWFIG: plt.show()



	### COMPUTE binned CROSS 21-kappa POWER SPECTRUM
			# >>> DO INTEGRAL OVER y MODES TO GET BINNED 21-KAPPA CROSS-CORRELATION SPECTRUM // THEN REMOVE kpar MODES
    K_21_kpar_arr = np.sin(y_arr_full*Deltanutilde/1.0)/(y_arr_full*Deltanutilde/1.0)
    K_21_kpar_2d_arr = np.outer(np.ones(n_ell_full), K_21_kpar_arr)

    Cl_21_kappa_cross_ell_zbin_arr= np.trapz(1.0/(2.0*np.pi) * Cl_21_kappa_cross_ell_y_2d_arr*K_21_kpar_2d_arr, x=y_arr_full, axis=1)
    Cl_21_kappa_cross_ell_zbin_cum_arr = scipy.integrate.cumtrapz(1.0/(2.0*np.pi) * Cl_21_kappa_cross_ell_y_2d_arr*K_21_kpar_2d_arr, x=y_arr_full, axis=1)
    Cl_21_noise_ell_zbin_arr=Cl_21_noise_ell_arr*np.trapz(1.0/(2.0*np.pi) * K_21_kpar_arr, x=y_arr_full, axis=0)

    k_cut_arr = np.array([1e-3,1e-2]); n_cut = np.size(k_cut_arr) ; ind_cut_arr=np.zeros(n_cut) # k_cut_arr: [1e-5,1e-4,1e-3,3e-3,1e-2]

    for i, kci in enumerate(k_cut_arr): ind_cut_arr[i]=np.int(np.max(np.where(y_arr_full/rnu_A <= kci)))
    ind_cut_arr_int = ind_cut_arr.astype(int)
    Cl_21_kappa_cross_ell_zbin_cum_reverse_arr = np.outer(Cl_21_kappa_cross_ell_zbin_arr,np.ones(n_cut)) - Cl_21_kappa_cross_ell_zbin_cum_arr[:,ind_cut_arr_int]
    fig, ax1 = plt.subplots()
    ax1.legend(loc='lower left')
    ax1.loglog(ell_arr_full,Cl_21_kappa_cross_ell_zbin_arr,linewidth=3.0,label='No cut')
    Clcross_plot_cut=np.abs(Cl_21_kappa_cross_ell_zbin_cum_reverse_arr[:,:])
    for i in np.arange(len(k_cut_arr)):
        y=k_cut_arr
        ax1.plot(ell_arr_full, Clcross_plot_cut[:,i],linewidth=3.0,label='kcut='+str(y[i]))
    ax1.legend(loc='upper left',ncol=5 ,fontsize=25)
    ax1.set_ylim([1e-18,1e-8])
    #ax1.set_xlim(15,7000)
    ax1.set_xlabel(r'$\ell$',fontsize=40); plt.ylabel(r'$C^{21\,\kappa}_\ell(k_{cut}; z_i) (mK)$',fontsize=50)
    ax1.tick_params(axis='both', which='major', labelsize=40)
    plt.tick_params(axis='both',labelsize=40)
    #plt.tight_layout()

    ax2 = plt.axes([0,0,1,1])
    ip = InsetPosition(ax1, [0.1,0.13,0.35,0.35])
    ax2.set_axes_locator(ip)
    #mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')
    ax2.plot(Kernel_arr0_1[0],Kernel_arr0_1[1],linewidth=3.0)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'k$_\parallel$ $[Mpc^{-1}]$',fontsize=30)
    ax2.set_ylabel(r'K$_\kappa(k_\parallel)$',fontsize=30)
    ax2.axvline(x=0.01,color='k',linestyle='--',linewidth=2)
    ax2.set_xlim([0,1e-1])
    ax2.set_ylim(1e-6,1e0)
    plt.tick_params(axis='both',labelsize=30)
    #plt.tight_layout()
    if SAVEFIG: plt.savefig(output_dir+'figure_HI_kappa_signal_noise_kcut_1d.png')
    if SHOWFIG: plt.show()

   # >>> Compute 1D cross SNR
    Cl_21_auto_ell_y_2d_arr=np.abs(Cl_21_auto_ell_y_zA(ell_arr_full, y_arr_full, zbin_prop, bias_var=1., gamma_var=1.))
    Cl21_bin = np.trapz(1.0/(2.0*np.pi) * Cl_21_auto_ell_y_2d_arr*K_21_kpar_2d_arr, x=y_arr_full, axis=1)
    Cl_var = np.sqrt(Cl_21_kappa_cross_ell_zbin_arr**2.+(Cl_21_noise_ell_zbin_arr+Cl21_bin)*(CMB_kappa_noise_ell+CMB_convergence_power(ell_arr_full)))
    Cl_var_Heather = np.sqrt(Cl_21_kappa_cross_ell_zbin_arr**2.+(Cl_21_noise_ell_arr+Cl21_bin)*(CMB_kappa_noise_ell+CMB_convergence_power(ell_arr_full)))

    delta_ell=np.array([])
    for i in np.arange(n_ell_full-1):
        delta_ell = np.append(delta_ell,ell_arr_full[i+1]-ell_arr_full[i])
    delta_ell=np.insert(delta_ell,0,0)
    Nmodes = np.sqrt((2*ell_arr_full + 1)*fsky*delta_ell)
    SNR_1D = Nmodes*Cl_21_kappa_cross_ell_zbin_arr/(Cl_var)
    SNR_1D_H = Nmodes*Cl_21_kappa_cross_ell_zbin_arr/(Cl_var_Heather)
    Cumulative_SNR_1D = np.sqrt(np.cumsum(SNR_1D**2.))
    plt.plot(ell_arr_full,np.sqrt(np.cumsum(SNR_1D_H**2.)),label='Heather')
    plt.plot(ell_arr_full,Cumulative_SNR_1D,label='Cross SNR')
    np.savetxt('Warren_datafiles/Cross_SNR_1D_zc_'+str(z_A)+'.dat',(ell_arr_full,Cumulative_SNR_1D))
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$SNR$ $z_c=$ ' + str(round(z_A,2)))
    plt.xlim(0,2000)
    plt.legend()
    if SHOWFIG: plt.show()

	# >>> COMPUTE 21-KAPPA SNR FOR FULL CORRELATION IN ell, y BINS

    kpar_arr = np.zeros(num_k_bins_y) ; kperp_arr= np.zeros(num_k_bins_ell) ; SNR_arr_21_kappa = np.zeros((num_k_bins_y,num_k_bins_ell))

    Cl_21_auto_interp_fn_full = scipy.interpolate.RectBivariateSpline(ell_arr_full, y_arr_full, Cl_21_auto_ell_y_2d_arr)

    Cl_21_kappa_cross_ell_y_2d_arr= np.abs(Cl_21_kappa_cross_ell_y(ell_arr_full, y_arr_full, zbin_prop, bias_var=1., gamma_var=1.))
    Cl_21_kappa_cross_interp_fn_full = scipy.interpolate.RectBivariateSpline(ell_arr_full, y_arr_full, Cl_21_kappa_cross_ell_y_2d_arr)

    Cl_21_noise_ell_arr=np.abs(HI_angular_noise_ell_y(ell_arr_full, zbin_prop, experiment_name='hirax', mode='interferometer', show_plots=False))
    Cl_21_noise_interp_fn_full = interp1d(ell_arr_full, Cl_21_noise_ell_arr,fill_value=1e20)

    CMB_kappa_noise_interp_fn_full = interp1d(ell_arr_full, CMB_convergence_noise(ell_arr_full))

    CMB_kappa_signal_interp_fn_full = interp1d(ell_arr_full, CMB_convergence_power(ell_arr_full))

    for bin_number_y in np.linspace(0,num_k_bins_y-1,num_k_bins_y):
		for bin_number_ell in np.linspace(0,num_k_bins_ell-1,num_k_bins_ell):
			kpar_bin_min, kpar_bin_max = y_min_full/rnu_A + delta_kpar*np.array([bin_number_y, bin_number_y+1])    # bin_number starts at 0
			kperp_bin_min, kperp_bin_max = ell_min_full/chi_A + delta_kperp*np.array([bin_number_ell, bin_number_ell+1])
			kpar_arr[np.int(bin_number_y)] = kpar_bin_min  ; kperp_arr[np.int(bin_number_ell)] = kperp_bin_min
			y_bin_arr = y_arr_full[(y_arr_full > kpar_bin_min*rnu_A) & (y_arr_full < kpar_bin_max*rnu_A)]
			ell_bin_arr = ell_arr_full[(ell_arr_full > kperp_bin_min*chi_A) & (ell_arr_full < kperp_bin_max*chi_A)]
			n_y_bin = y_bin_arr.size
			Cl_21_auto_ell_y_2d_bin_arr = Cl_21_auto_interp_fn_full(ell_bin_arr,y_bin_arr)
			Cl_21_kappa_cross_ell_y_2d_arr= Cl_21_kappa_cross_interp_fn_full(ell_bin_arr, y_bin_arr)
			ell_bin_2d_arr  = np.outer(ell_bin_arr,np.ones(n_y_bin))

			Cl_21_noise_ell_y_2d_bin_arr = np.outer(Cl_21_noise_interp_fn_full(ell_bin_arr),np.ones(n_y_bin))
			Cl_21_noise_ell_y_2d_bin_arr_allsky =Cl_21_noise_ell_y_2d_bin_arr# HI_angular_noise_ell_y_allsky(Cl_21_noise_ell_y_2d_bin_arr)
			CMB_kappa_noise_ell_y_2d_bin_arr = np.outer(CMB_kappa_noise_interp_fn_full(ell_bin_arr),np.ones(n_y_bin))
			CMB_kappa_signal_ell_y_2d_bin_arr = np.outer(CMB_kappa_signal_interp_fn_full(ell_bin_arr),np.ones(n_y_bin))

			Variance_21_kappa_ell_y_2d_arr = Cl_21_kappa_cross_ell_y_2d_arr**2. + (np.abs(Cl_21_noise_ell_y_2d_bin_arr_allsky)+Cl_21_auto_ell_y_2d_bin_arr)*(CMB_kappa_noise_ell_y_2d_bin_arr+CMB_kappa_signal_ell_y_2d_bin_arr)

			SN_ratio_2d_arr =  Cl_21_kappa_cross_ell_y_2d_arr/np.sqrt(np.abs(Variance_21_kappa_ell_y_2d_arr))
			SNR_sq_21_kappa = Mode_Volume_Factor/(2*np.pi**2) * np.trapz(np.trapz(SN_ratio_2d_arr**2*ell_bin_2d_arr, ell_bin_arr, axis=0), y_bin_arr, axis=0)
			SNR_arr_21_kappa[np.int(bin_number_y),np.int(bin_number_ell)] = np.sqrt(np.abs(SNR_sq_21_kappa))

    pylab.pcolormesh(kperp_arr/0.678,kpar_arr/0.678,np.abs(SNR_arr_21_kappa)) ;cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    #plt.yscale('log')
    pylab.ylim([np.min(kpar_arr/0.678),np.max(kpar_arr/0.678)]) ; pylab.xlim([np.min(kperp_arr/0.678),np.max(kperp_arr/0.678)])
    plt.ylabel(r'$k_\parallel$ $[hMpc^{-1}]$',fontsize=15); plt.xlabel(r'$k_\perp$ $[hMpc^{-1}]$',fontsize=15); #plt.title('Pixel SN for 21-$\kappa$', x=1.13, y=1.05,fontsize=15)
    plt.tight_layout()
    if SAVEFIG: pylab.savefig(output_dir+'figure_HI_kappa_SNR_2d.png')
    if SHOWFIG: pylab.show()
    snr_ell=np.array([])
    for i in np.arange(num_k_bins_ell):
        snr_ell = np.append(snr_ell,np.sqrt(np.sum(SNR_arr_21_kappa[:,i]**2.)))


    #### 2D  kpar cut
    for bin_number_y in np.linspace(0,num_k_bins_y-1,num_k_bins_y):
		for bin_number_ell in np.linspace(0,num_k_bins_ell-1,num_k_bins_ell):
			kpar_bin_min, kpar_bin_max = kcut + delta_kpar*np.array([bin_number_y, bin_number_y+1])    # bin_number starts at 0
			kperp_bin_min, kperp_bin_max = ell_min_full/chi_A + delta_kperp*np.array([bin_number_ell, bin_number_ell+1])
			kpar_arr[np.int(bin_number_y)] = kpar_bin_min  ; kperp_arr[np.int(bin_number_ell)] = kperp_bin_min
			y_bin_arr = y_arr_full[(y_arr_full > kpar_bin_min*rnu_A) & (y_arr_full < kpar_bin_max*rnu_A)]
			ell_bin_arr = ell_arr_full[(ell_arr_full > kperp_bin_min*chi_A) & (ell_arr_full < kperp_bin_max*chi_A)]
			n_y_bin = y_bin_arr.size
			Cl_21_auto_ell_y_2d_bin_arr = Cl_21_auto_interp_fn_full(ell_bin_arr,y_bin_arr)
			Cl_21_kappa_cross_ell_y_2d_arr= Cl_21_kappa_cross_interp_fn_full(ell_bin_arr, y_bin_arr)
			ell_bin_2d_arr  = np.outer(ell_bin_arr,np.ones(n_y_bin))

			Cl_21_noise_ell_y_2d_bin_arr = np.outer(Cl_21_noise_interp_fn_full(ell_bin_arr),np.ones(n_y_bin))
			Cl_21_noise_ell_y_2d_bin_arr_allsky =Cl_21_noise_ell_y_2d_bin_arr# HI_angular_noise_ell_y_allsky(Cl_21_noise_ell_y_2d_bin_arr)
			CMB_kappa_noise_ell_y_2d_bin_arr = np.outer(CMB_kappa_noise_interp_fn_full(ell_bin_arr),np.ones(n_y_bin))
			CMB_kappa_signal_ell_y_2d_bin_arr = np.outer(CMB_kappa_signal_interp_fn_full(ell_bin_arr),np.ones(n_y_bin))

			Variance_21_kappa_ell_y_2d_arr = Cl_21_kappa_cross_ell_y_2d_arr**2. + (np.abs(Cl_21_noise_ell_y_2d_bin_arr_allsky)+Cl_21_auto_ell_y_2d_bin_arr)*(CMB_kappa_noise_ell_y_2d_bin_arr+CMB_kappa_signal_ell_y_2d_bin_arr)

			SN_ratio_2d_arr =  Cl_21_kappa_cross_ell_y_2d_arr/np.sqrt(np.abs(Variance_21_kappa_ell_y_2d_arr))
			SNR_sq_21_kappa = Mode_Volume_Factor/(2*np.pi**2) * np.trapz(np.trapz(SN_ratio_2d_arr**2*ell_bin_2d_arr, ell_bin_arr, axis=0), y_bin_arr, axis=0)
			SNR_arr_21_kappa[np.int(bin_number_y),np.int(bin_number_ell)] = np.sqrt(np.abs(SNR_sq_21_kappa))

    pylab.pcolormesh(kperp_arr/0.678,kpar_arr/0.678,np.abs(SNR_arr_21_kappa))
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    pylab.ylim([np.min(kpar_arr/0.678),np.max(kpar_arr/0.678)]) ; pylab.xlim([np.min(kperp_arr/0.678),np.max(kperp_arr/0.678)])
    plt.ylabel(r'$k_\parallel$ $[hMpc^{-1}]$',fontsize=15); plt.xlabel(r'$k_\perp$ $[hMpc^{-1}]$',fontsize=15);# plt.title('Pixel SN for 21-$\kappa$ $k_\parallel cut:$'+str(kcut), x=1.0, y=1.05,fontsize=15)
    #if SAVEFIG: pylab.savefig(output_dir+'figure_HI_kappa_SNR_2d.png')
    #plt.yscale('log')
    plt.tight_layout()
    if SHOWFIG: pylab.show()

    snr_ell_cut=np.array([])
    for i in np.arange(num_k_bins_ell):
        snr_ell_cut = np.append(snr_ell_cut,np.sqrt(np.sum(SNR_arr_21_kappa[:,i]**2.)))

    SNRcutmin3 = np.loadtxt('CrossSNR_cumsum_k0.01.dat')
    fig, ax1 = plt.subplots()
    ax3 = ax1.twinx()
    ax1.legend(loc='lower left')
    ax1.loglog(ell_arr_full,Cl_21_kappa_cross_ell_zbin_arr,linewidth=3.0,label='No cut')
    Clcross_plot_cut=np.abs(Cl_21_kappa_cross_ell_zbin_cum_reverse_arr[:,:])
    for i in np.arange(len(k_cut_arr)):
        y=k_cut_arr
        ax1.plot(ell_arr_full, Clcross_plot_cut[:,i],linewidth=3.0,label='$k_{\parallel,cut}$='+str(y[i]))
    ax1.legend(loc='upper left',ncol=5 ,fontsize=35)
    ax3.loglog(kperp_arr*chi_A,np.sqrt(np.cumsum(snr_ell**2.)),'C0--',linewidth=3.0)

    ax3.loglog(kperp_arr*chi_A,np.sqrt(np.cumsum(snr_ell_cut**2.)),'C1--',linewidth=3.0)
    ax3.loglog(SNRcutmin3[0],SNRcutmin3[1],'C2--',linewidth=3.0)

    ax1.set_ylim([1e-15,1e-8])
    ax1.set_xlim(85,478)
    ax1.set_xscale('linear')
    ax1.set_xlabel(r'$\ell$',fontsize=50); ax1.set_ylabel(r'$C^{21\,\kappa}_\ell(k_{\parallel,cut}; z_i) (mK)$',fontsize=50)
    ax3.set_ylabel('SNR',fontsize=50)
    ax1.spines["bottom"].set_linewidth(3)
    ax1.spines["top"].set_linewidth(3)
    ax1.spines["left"].set_linewidth(3)
    ax1.spines["right"].set_linewidth(3)


    ax1.tick_params(axis='both', which='major', labelsize=40)
    plt.tick_params(axis='both',labelsize=40)

    #plt.tight_layout()

    ax2 = plt.axes([0,0,1,1])
    ip = InsetPosition(ax1, [0.63,0.15,0.33,0.33]) #left, bottom, width, height:0.1,0.13,0.35,0.35
    ax2.set_axes_locator(ip)
    #mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')
    ax2.plot(Kernel_arr0_1[0],Kernel_arr0_1[1],'k',linewidth=3.0)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'k$_\parallel$ $[Mpc^{-1}]$',fontsize=40)
    ax2.set_ylabel(r'|K$_\kappa(k_\parallel)|$',fontsize=40)
    ax2.axvline(x=0.01,color='r',linestyle='--',linewidth=2)
    ax2.set_xlim([0,1e-1])
    ax2.set_ylim(1e-6,1e0)
    ax2.spines["bottom"].set_linewidth(3)
    ax2.spines["top"].set_linewidth(3)
    ax2.spines["left"].set_linewidth(3)
    ax2.spines["right"].set_linewidth(3)
    plt.tick_params(axis='both',labelsize=30)
    ax2.tick_params(axis='both', which='major',labelsize=40)
    #ax3.tick_params(axis='both',labelsize=30)
    #plt.tight_layout()
    if SAVEFIG: plt.savefig(output_dir+'figure_HI_kappa_signal_noise_kcut_1d.png')
    if SHOWFIG: plt.show()
    #plt.show()


def compute_SNR_Cl_21_21_auto(zbin_prop, delta_kpar = 0.01,  delta_kperp = 0.01,
                              num_k_bins_ell = 15,  num_k_bins_y=14,T_obs = 2.):

	## SET Z BIN PROPERTIES

    Deltanutilde = zbin_prop.Deltanutilde    # Bin width

    z_A = zbin_prop.z_A

    nutilde_A = zbin_prop.nutilde_A
    chi_A=zbin_prop.chi_A
    rnu_A=zbin_prop.rnu_A
    FOV_A_sr = zbin_prop.FOV_A_sr

	### SET RADIAL MODES
    y_min = zbin_prop.y_min_expt ; y_max = zbin_prop.knl*rnu_A#zbin_prop.y_max_expt#zbin_prop.knl*rnu_A#
    result = zbin_prop.get_y_arr_expt() ; n_y = result[0]  ; y_arr = result[1]

	### SET TRANSVERSE MODES
    ell_min = zbin_prop.ell_min_expt  ; ell_max = zbin_prop.knl*chi_A#zbin_prop.ell_max_expt #zbin_prop.knl*chi_A#
    result = zbin_prop.get_ell_arr_expt() ; n_ell = result[0]  ; ell_arr = result[1]
    N_patches = zbin_prop.N_patches
    Mode_Volume_Factor = zbin_prop.Mode_Volume_Factor
	### COMPUTE AUTO 21-21 POWER SPECTRUM AND SIGNAL-TO-NOISE

    K_21_kpar_arr = np.sin(y_arr*Deltanutilde/1.0)/(y_arr*Deltanutilde/1.0)
    K_21_kpar_2d_arr = np.outer(np.ones(n_ell), K_21_kpar_arr)

    ell_full_arr = np.linspace(10,3000,3000)
    Cl_21_auto_full = Cl_21_auto_ell_y_zA(ell_full_arr, np.array([100,1000,2000,3000,4000]), zbin_prop, bias_var=1., gamma_var=1.)
    Cl_21_noise_full =HI_angular_noise_ell_y(ell_full_arr, zbin_prop, T_obs = T_obs, experiment_name='hirax', mode='interferometer', show_plots=False)
    Cl_21_noise_full_allsky = HI_angular_noise_ell_y_allsky(Cl_21_noise_full)/np.sqrt(fsky * ell_full_arr)
    plt.loglog(ell_full_arr,Cl_21_noise_full_allsky,'-.')
    plt.loglog(ell_full_arr,Cl_21_auto_full[:,0],label='y=100')
    plt.loglog(ell_full_arr,Cl_21_auto_full[:,1],label='y=1000')
    plt.loglog(ell_full_arr,Cl_21_auto_full[:,2],label='y=2000')
    plt.loglog(ell_full_arr,Cl_21_auto_full[:,3],label='y=3000')
    plt.loglog(ell_full_arr,Cl_21_auto_full[:,4],label='y=4000')
    plt.xlim(10,3000)
    plt.ylim(1e-12,1e-8)
    plt.xlabel('$\ell$',fontsize=15)
    plt.ylabel('$C^{21}_\ell(y) \quad vs \quad N^{21}_\ell(y) \,\, (mK)^2$',fontsize=15)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both',labelsize=15)
    plt.tight_layout()
    plt.show()

    Kernel_arr0_1 = np.loadtxt('Full_HIRAX_Lensing_Kernel_kpar.dat')
    kkkd = interp1d(Kernel_arr0_1[0], Kernel_arr0_1[1],bounds_error=False,fill_value=1e-8)

    K_2d = np.outer(np.ones(3000),kkkd(y_arr/rnu_A))

    ell_arr_test = np.linspace(0.0001*chi_A,0.3*chi_A,100)
    y_arr_test = np.linspace(0.0001*rnu_A,0.3*rnu_A,100)
    Cl_21_test2d = Cl_21_auto_ell_y_zA(ell_arr_test, y_arr_test, zbin_prop, bias_var=1., gamma_var=1.)
    plt.loglog(Cl_21_test2d*chi_A**2.*rnu_A)

    Cl_21_auto_ell_y_2d_arr=Cl_21_auto_ell_y_zA(ell_arr, y_arr, zbin_prop, bias_var=1., gamma_var=1.) # 21cm angular PS at z vs ell,y
    Cl_21_noise_ell_arr=HI_angular_noise_ell_y(ell_arr, zbin_prop, T_obs = 4,  experiment_name='hirax', mode='interferometer', show_plots=False)
    Cl_21_noise_ell_y_2d_arr=  np.outer(Cl_21_noise_ell_arr,np.ones(n_y))
    ell_2d_arr=  np.outer(ell_arr,np.ones(n_y))

    Cl_21_auto_ell_y_2d_arr=np.abs(Cl_21_auto_ell_y_zA(ell_arr, y_arr, zbin_prop, bias_var=1., gamma_var=1.))
    Cl21_bin = np.trapz(1.0/(2.0*np.pi) * Cl_21_auto_ell_y_2d_arr*K_21_kpar_2d_arr, x=y_arr, axis=1)
    Cl_21_noise_ell_zbin_arr=Cl_21_noise_ell_arr*np.trapz(1.0/(2.0*np.pi) * K_21_kpar_arr, x=y_arr, axis=0)
    plt.loglog(ell_arr,Cl_21_noise_ell_zbin_arr/np.sqrt(fsky * ell_arr),'-.')
    plt.loglog(ell_arr,Cl21_bin)
    plt.ylabel(r'$C^{21}_{\ell,bin} \quad vs \quad N^{21}_\ell/ \sqrt{f_{sky} \,\ell}$ $(mK)^2$')
    if SHOWFIG: plt.show()

    ###>>> 1D SNR
    delta_ell=np.array([])
    for i in np.arange(n_ell-1):
        delta_ell = np.append(delta_ell,ell_arr[i+1]-ell_arr[i])
    delta_ell=np.insert(delta_ell,0,0)
    Nmodes = np.sqrt((2*ell_arr + 1)*fsky*delta_ell)
    SNR_1D = Nmodes*Cl21_bin/(Cl21_bin+Cl_21_noise_ell_zbin_arr)
    Cumulative_SNR_1D = np.sqrt(np.cumsum(SNR_1D**2.))

    Cl21_plot= Cl_21_auto_ell_y_2d_arr[:,0:n_y/5:100]

    Cl_21_auto_interp_fn = scipy.interpolate.RectBivariateSpline(ell_arr, y_arr, Cl_21_auto_ell_y_2d_arr)
    Cl_21_noise_interp_fn = scipy.interpolate.RectBivariateSpline(ell_arr, y_arr, Cl_21_noise_ell_y_2d_arr)

    delta_y_plot = delta_kpar * rnu_A ; delta_ell_plot = delta_kperp * chi_A

    Cl_21_noise_ell_y_2d_arr_allsky = HI_angular_noise_ell_y_allsky(Cl_21_noise_ell_y_2d_arr)
    SN_sq_ratio = 0.5*delta_y_plot*delta_ell_plot*Mode_Volume_Factor/(2*np.pi**2) * ell_2d_arr*(Cl_21_auto_ell_y_2d_arr/(Cl_21_noise_ell_y_2d_arr_allsky+Cl_21_auto_ell_y_2d_arr))**2

    PLOT_OPTION = 'LINEAR' # 'LOG' # 'LINEAR'
    if PLOT_OPTION == 'LOG':
		Y_plot_arr = np.log10(y_arr/rnu_A) ; L_plot_arr = np.log10(ell_arr/chi_A)
		ell_axis_arr=np.log10(ell_arr); y_axis_arr=np.log10(y_arr)
		ell_label_arr = '$\log(k_\perp = \ell/{\chi_\parallel})$'; y_label_arr='$\log(k_\parallel=  y/r_\\nu$)'
		ell_2_label_arr = '$\log(\ell)$' ; y_2_label_arr = '$\log(y)$'
		ell_axis_max = np.max(ell_axis_arr); y_axis_max = np.max(y_axis_arr)

    if PLOT_OPTION == 'LINEAR':
		Y_plot_arr = (y_arr/rnu_A) ; L_plot_arr = (ell_arr/chi_A)
		ell_axis_arr=(ell_arr); y_axis_arr=(y_arr)
		ell_label_arr = '$k_\perp = \ell/{\chi_\parallel}$'; y_label_arr='$k_\parallel=  y/r_\\nu$'
		ell_axis_max = 0.15 * chi_A; y_axis_max = 0.15 * rnu_A
		ell_2_label_arr = '$\ell$' ; y_2_label_arr = '$y$'

    Y,L = np.meshgrid(Y_plot_arr,L_plot_arr) ; fig, axs = plt.subplots()
    cs = axs.contourf(Y,L,np.sqrt(SN_sq_ratio),15) ; fig.colorbar(cs, ax=axs, pad=0.1)
    plt.xlabel(y_label_arr); plt.ylabel(ell_label_arr); plt.title('Pixel SN', x=1.13, y=1.05)

    if PLOT_OPTION == 'LINEAR':
		axs.set_ylim(np.min(L),0.15); axs.set_xlim(np.min(Y),0.15)

    ax2 = axs.twinx(); ax2.set_ylim(np.min(ell_axis_arr),ell_axis_max); ax2.set_ylabel(ell_2_label_arr)

    ax3 = axs.twiny(); ax3.set_xlim(np.min(y_axis_arr),y_axis_max); ax3.set_xlabel(y_2_label_arr)

    if SHOWFIG: plt.show()

	#num_k_bins = 15
    num_k_bins_y = num_k_bins_ell = 20
    kpar_arr = np.zeros(num_k_bins_y) ; kperp_arr= np.zeros(num_k_bins_ell) ; SNR_arr = np.zeros((num_k_bins_y,num_k_bins_ell))
    kpar_min = zbin_prop.kpar_min
    kperp_min = zbin_prop.kperp_min

    for bin_number_y in np.linspace(0,num_k_bins_y-1,num_k_bins_y):
		for bin_number_ell in np.linspace(0,num_k_bins_ell-1,num_k_bins_ell):
			kpar_bin_min, kpar_bin_max = kpar_min + delta_kpar*np.array([bin_number_y, bin_number_y+1])    # bin_number starts at 0
			kperp_bin_min, kperp_bin_max = kperp_min + delta_kperp*np.array([bin_number_ell, bin_number_ell+1])

			kpar_arr[np.int(bin_number_y)] = kpar_bin_min  ; kperp_arr[np.int(bin_number_ell)] = kperp_bin_min

			y_bin_arr = y_arr[(y_arr > kpar_bin_min*rnu_A) & (y_arr < kpar_bin_max*rnu_A)]
			ell_bin_arr = ell_arr[(ell_arr > kperp_bin_min*chi_A) & (ell_arr < kperp_bin_max*chi_A)]
			n_y_bin = y_bin_arr.size

			ell_bin_2d_arr  = np.outer(ell_bin_arr,np.ones(n_y_bin))

			Cl_21_auto_ell_y_2d_bin_arr = Cl_21_auto_interp_fn(ell_bin_arr,y_bin_arr)
			Cl_21_noise_ell_y_2d_bin_arr = Cl_21_noise_interp_fn(ell_bin_arr,y_bin_arr)

			Cl_21_noise_ell_y_2d_bin_arr_allsky = HI_angular_noise_ell_y_allsky(Cl_21_noise_ell_y_2d_bin_arr)

			SN_ratio_2d_arr =  Cl_21_auto_ell_y_2d_bin_arr/(Cl_21_noise_ell_y_2d_bin_arr_allsky + Cl_21_auto_ell_y_2d_bin_arr)

			SNR_sq = 0.5 * Mode_Volume_Factor/(2*np.pi**2) * np.trapz(np.trapz(SN_ratio_2d_arr**2*ell_bin_2d_arr, ell_bin_arr, axis=0), y_bin_arr, axis=0)
			SNR_arr[np.int(bin_number_y),np.int(bin_number_ell)] = np.sqrt(SNR_sq)


    snr_ell=np.array([])
    for i in np.arange(num_k_bins_ell):
        snr_ell = np.append(snr_ell,np.sqrt(np.sum(SNR_arr[:,i]**2.)))
    pylab.pcolormesh(kperp_arr,kpar_arr,SNR_arr) ;  cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.tick_params(axis='both', which='major', labelsize=13)
    pylab.xlim([np.min(kperp_arr),np.max(kperp_arr)])
    pylab.ylim([np.min(kpar_arr),np.max(kpar_arr)])
    plt.xlabel(r'$k_\perp$ $[Mpc^{-1}]$',fontsize=16); plt.ylabel(r'$k_\parallel$ $[Mpc^{-1}]$',fontsize=16); #plt.title('Pixel SN for 21-21', x=1.13, y=1.05)
    plt.tight_layout()
    plt.title('HI-HI SNR',fontsize=16)
    if SAVEFIG: pylab.savefig(output_dir+'figure_HI_SNR_2d.png')
    if SHOWFIG: pylab.show()
    print '21-21 SNR', np.sqrt(np.sum(SNR_arr**2))
    return np.sqrt(np.sum(SNR_arr**2))


def compute_SNR_Bl_21_21_kappa_cross(zbin_prop, delta_kpar = 0.01, delta_kperp = 0.01,
                                     num_k_bins_ell = 15, num_k_bins_y=15,T_obs = 2.):

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

	### COMPUTE 21-21-kappa INTEGRATED BISPECTRUM and SIGNAL-TO-NOISE

    n_ell_coarse = 999 ; n_y_coarse = 1000
    ell_min_coarse = ell_min ; y_min_coarse = y_min
    knl = zbin_prop.knl
    ell_max_coarse = knl*chi_A#0.01*num_k_bins_ell*chi_A + ell_min_coarse ; # nonlinear_factor = 17*(chi_A/3396.21097) ;
    y_max_coarse = knl*rnu_A#0.01*num_k_bins_y*rnu_A + y_min_coarse
    ell_coarse_arr = np.linspace(ell_min_coarse,ell_max_coarse,n_ell_coarse); y_coarse_arr = np.linspace(y_min_coarse,y_max_coarse,n_y_coarse)
    Cl_21_auto_ell_y_2d_arr=Cl_21_auto_ell_y_zA(ell_coarse_arr, y_coarse_arr, zbin_prop, bias_var=1., gamma_var=1.) # 21cm angular PS at z vs ell,y
    Cl_21_noise_ell_arr=HI_angular_noise_ell_y(ell_coarse_arr, zbin_prop,T_obs = 4, experiment_name='hirax', mode='interferometer', show_plots=False)
    Cl_21_noise_ell_y_2d_arr=  np.outer(Cl_21_noise_ell_arr,np.ones(n_y_coarse))


    B_ell_y_zA = integ_bispec_kappa_21_21_ell_y_zA(ell_coarse_arr, y_coarse_arr, zbin_prop, bias_var=1., gamma_var=1.)

    C_21_21_kappa_zA=B_ell_y_zA


	# Compute Average CMB kappa variance

    CMB_kappa_average_variance_sky = compute_average_kappa_variance_sky_zA(zbin_prop, bias_var=1., gamma_var=1.)


    Cl_21_noise_ell_arr_allsky=(1./fsky) * HI_angular_noise_ell_y(ell_coarse_arr, zbin_prop,T_obs = T_obs, experiment_name='hirax', mode='interferometer', show_plots=False)
    CMB_kappa_noise_ell = CMB_convergence_noise(ell_coarse_arr)
    expt_noise_term_ell_arr = np.sqrt( 3*(fsky/N_patches)**2 * Cl_21_noise_ell_arr_allsky**2 * CMB_kappa_noise_ell)
    CMB_kappa_signal_ell = CMB_convergence_power(ell_coarse_arr)


    plt.loglog(ell_coarse_arr,expt_noise_term_ell_arr/np.sqrt(ell_coarse_arr),'-.')
    Bispec_plot = C_21_21_kappa_zA[:,10:(n_y_coarse-1)/2:100]
    for i in np.arange(len(y_coarse_arr[10:n_y_coarse/2:100])):
        y=y_coarse_arr[10:n_y_coarse/2:100]
        plt.plot(ell_coarse_arr, Bispec_plot[:,i],label='y='+str(round(y[i],2)))
    plt.legend()

    plt.xlabel('$\ell$',fontsize=15); plt.ylabel('$B^{21\,21\,\kappa}_{\ell, patch}(y) \quad vs \quad \sqrt{3} \, Ne^{21}_\ell(y)\sqrt{N^{\kappa}_\ell} \,f_{patch} \,\, (mK)^2$',fontsize=14)

    if SAVEFIG: pylab.savefig(output_dir+'figure_HI_HI_kappa_signal_noise_1d.png')
    if SHOWFIG: plt.show()

	# COMPUTE 21-21-Kappa Cross SNR IN BINS of kpar,kperp = 0.01/Mpc
    num_k_bins_y = num_k_bins_ell = 20

    kpar_arr = np.zeros(num_k_bins_y) ; kperp_arr= np.zeros(num_k_bins_ell) ; SNR_arr = np.zeros((num_k_bins_y,num_k_bins_ell))
    kpar_min = zbin_prop.kpar_min
    kperp_min = zbin_prop.kperp_min

    C_21_21_kappa_zA_auto_interp_fn = scipy.interpolate.RectBivariateSpline(ell_coarse_arr, y_coarse_arr, np.abs(C_21_21_kappa_zA))
    C_21_21_zA_noise_interp_fn = scipy.interpolate.RectBivariateSpline(ell_coarse_arr, y_coarse_arr, np.abs(Cl_21_noise_ell_y_2d_arr)) # Needs noise computed before
    C_21_21_zA_auto_interp_fn  = scipy.interpolate.RectBivariateSpline(ell_coarse_arr, y_coarse_arr, np.abs(Cl_21_auto_ell_y_2d_arr))


    for bin_number_y in np.linspace(0,num_k_bins_y-1,num_k_bins_y):
		for bin_number_ell in np.linspace(0,num_k_bins_ell-1,num_k_bins_ell):
			kpar_bin_min, kpar_bin_max = kpar_min + delta_kpar*np.array([bin_number_y, bin_number_y+1])    # bin_number starts at 0
			kperp_bin_min, kperp_bin_max = kperp_min + delta_kperp*np.array([bin_number_ell, bin_number_ell+1])

			kpar_arr[np.int(bin_number_y)] = kpar_bin_min  ; kperp_arr[np.int(bin_number_ell)] = kperp_bin_min

			y_bin_arr = y_coarse_arr[(y_coarse_arr > kpar_bin_min*rnu_A) & (y_coarse_arr < kpar_bin_max*rnu_A)]
			ell_bin_arr = ell_coarse_arr[(ell_coarse_arr > kperp_bin_min*chi_A) & (ell_coarse_arr < kperp_bin_max*chi_A)]

			n_y_bin = y_bin_arr.size

			ell_bin_2d_arr  = np.outer(ell_bin_arr,np.ones(n_y_bin))

			C_21_21_kappa_zA_ell_y_2d_bin_arr = C_21_21_kappa_zA_auto_interp_fn(ell_bin_arr,y_bin_arr)
			Cl_21_noise_ell_y_2d_bin_arr = C_21_21_zA_noise_interp_fn(ell_bin_arr,y_bin_arr)
			Cl_21_auto_ell_y_2d_bin_arr  = C_21_21_zA_auto_interp_fn(ell_bin_arr,y_bin_arr)


			CMB_kappa_noise_ell = CMB_convergence_noise(ell_bin_2d_arr)
			CMB_kappa_signal_ell = CMB_convergence_power(ell_bin_2d_arr)

			Cl_21_noise_ell_y_2d_bin_arr_allsky = np.abs(Cl_21_noise_ell_y_2d_bin_arr)/fsky

			Variance_21_21_kappa_2d_arr	= 6* N_patches**2 * C_21_21_kappa_zA_ell_y_2d_bin_arr**2 + \
			(3*fsky**2)*(Cl_21_noise_ell_y_2d_bin_arr_allsky + Cl_21_auto_ell_y_2d_bin_arr)**2 * CMB_kappa_average_variance_sky


			SN_sq_ratio_2d_arr = N_patches**2 * C_21_21_kappa_zA_ell_y_2d_bin_arr**2/Variance_21_21_kappa_2d_arr


			SNR_sq = 0.5 * Mode_Volume_Factor/(2*np.pi**2) * np.trapz(np.trapz(SN_sq_ratio_2d_arr*ell_bin_2d_arr, ell_bin_arr, axis=0), y_bin_arr, axis=0)

			SNR_arr[np.int(bin_number_y),np.int(bin_number_ell)] = np.sqrt(SNR_sq)


    pylab.pcolormesh(kperp_arr,kpar_arr,SNR_arr) ;  cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel(r'$k_\perp$ $[Mpc^{-1}]$',fontsize=16); plt.ylabel(r'$k_\parallel$ $[Mpc^{-1}]$',fontsize=16)#; plt.title('Bispectrum Pixel SN', x=1.13, y=1.05,fontsize=15)
    plt.tight_layout()
    plt.title('HI-HI-$\kappa$ SNR',fontsize=16)
    if SAVEFIG: pylab.savefig(output_dir+'figure_HI_HI_kappa_SNR_2d.png')
    if SHOWFIG: pylab.show()
    print '21-21-kappa SNR', np.sqrt(np.sum(SNR_arr**2))

    plt.close(); pylab.close();


def plot_ellipse_sub(twod_mean, twod_cov_matrix, twod_label, figsub):
    figsub.set_xlabel(twod_label[0],fontsize=10)
    figsub.set_ylabel(twod_label[1],fontsize=10)
    plot_kwargs_b = {'color':'b','linestyle':'-','linewidth':3,'alpha':0.8} ;	fill_kwargs_b = {'color':'b','alpha':0.5}
    plot_kwargs_r = {'color':'r','linestyle':'-','linewidth':3,'alpha':0.8} ;	fill_kwargs_r = {'color':'r','alpha':0.5}
    plot_ellipse(x_cent=twod_mean[0], y_cent=twod_mean[1], ax = figsub, cov=twod_cov_matrix, mass_level=0.67, plot_kwargs=plot_kwargs_b,fill=True,fill_kwargs=fill_kwargs_b)
    plot_ellipse(x_cent=twod_mean[0], y_cent=twod_mean[1], ax = figsub, cov=twod_cov_matrix, mass_level=0.95, plot_kwargs=plot_kwargs_r,fill=True,fill_kwargs=fill_kwargs_r)


def plot_cov_ellipses(lbl, zA, param_mean, param_Fisher_matrix, param_label):

    param_cov = scipy.linalg.inv(param_Fisher_matrix)

    fig = plt.figure()
    for i in range(param_mean.size):
       for j in range(param_mean.size):
           if i < j:
               figsub=plt.subplot2grid((param_mean.size,param_mean.size), (j,i))
               sig = np.sqrt(np.abs(param_cov[i,i]))
               twod_mean=np.array([param_mean[i],param_mean[j]])
               twod_label=np.array([param_label[i],param_label[j]])
               twod_cov_matrix = np.array([[param_cov[i,i],param_cov[i,j]], [param_cov[j,i],param_cov[j,j]]])
               plot_ellipse_sub(twod_mean, twod_cov_matrix, twod_label, figsub)
               if j != param_mean.size-1 :
                  figsub.set_xticklabels([])
		  figsub.tick_params(axis='both', which='major', labelsize=26)
               if i !=0  :
                  figsub.set_yticklabels([])
                  figsub.set_yticks([])
                  figsub.set_ylabel(str( ))
		  figsub.tick_params(axis='both', which='major', labelsize=26)
	       else:
                  figsub.set_ylabel(param_label[j],fontsize=35)
               #figsub.set_xlim(param_mean[i]- 0.05,param_mean[i]+0.05)
               #if j == param_mean.size-1:
                   #figsub.set_ylim(param_mean[j]-1,param_mean[j]+1)
               #else:
                   #figsub.set_ylim(param_mean[j]-0.05,param_mean[j]+0.05)

           elif i==j:
               figsub=plt.subplot2grid((param_mean.size,param_mean.size), (j,i))
               sig = np.sqrt(np.abs(param_cov[i,i]))
               xx = np.linspace(param_mean[i]-2.*sig, param_mean[i]+2.*sig, 4000)
               yy = 1./np.sqrt(2.*np.pi*sig**2.) * np.exp(-0.5 * ((xx-param_mean[i])/sig)**2.)
               yy /= np.max(yy)
               figsub.plot(xx, yy, ls='solid', color='red', lw=1.5) #colours[k][0]
               figsub.set_xlabel(param_label[i],fontsize=35)
               if i is 0:
                  fig.legend()
		  figsub.tick_params(axis='both', which='major', labelsize=26)
               if i != param_mean.size-1:
                  figsub.set_xticklabels([])
		  figsub.tick_params(axis='both', which='major', labelsize=26)
               figsub.set_yticks([])
           #figsub.set_xlabel(param_label[i],fontsize=25)
           figsub.tick_params(axis='both', which='major', labelsize=26)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tick_params(axis='both', which='major', labelsize=26)
    #fig.tight_layout()
    if SAVEFIG: plt.savefig(output_dir + lbl + '_z_'+np.str(zA)+'_Fisher_matrix.png')
    plt.show()

def plot_multiple_ellipse_sub(twod_mean, twod_cov_matrix, twod_label, figsub, Fisher_index, color=None):
    figsub.set_xlabel(twod_label[0],fontsize=22) #22
    figsub.set_ylabel(twod_label[1],fontsize=22) #22
    if color is None:
        plot_kwargs_b = {'color':'C'+str(Fisher_index),'linestyle':'-','linewidth':2.0,'alpha':0.9} ;	fill_kwargs_b = {'color':'C'+str(Fisher_index),'alpha':0.3}
        plot_kwargs_r = {'color':'C'+str(Fisher_index),'linestyle':'-','linewidth':2.0,'alpha':0.9} ;	fill_kwargs_r = {'color':'C'+str(Fisher_index),'alpha':0.3}
    else:
        plot_kwargs_b = {'color':color,'linestyle':'-','linewidth':2.0,'alpha':0.9} ;	fill_kwargs_b = {'color':color,'alpha':0.3}
        plot_kwargs_r = {'color':color,'linestyle':'-','linewidth':2.0,'alpha':0.9} ;	fill_kwargs_r = {'color':color,'alpha':0.3}

    plot_ellipse(x_cent=twod_mean[0], y_cent=twod_mean[1], ax = figsub, cov=twod_cov_matrix, mass_level=0.67, plot_kwargs=plot_kwargs_b,fill=True,fill_kwargs=fill_kwargs_b)
    plot_ellipse(x_cent=twod_mean[0], y_cent=twod_mean[1], ax = figsub, cov=twod_cov_matrix, mass_level=0.95, plot_kwargs=plot_kwargs_r,fill=True,fill_kwargs=fill_kwargs_r)

def plot_multiple_cov_ellipses(lbl, zA, param_mean, Fisher_list, list_names, param_label, colors=None):


    fig = plt.figure()
    for i in range(param_mean.size):
       for j in range(param_mean.size):
           if i < j:
               figsub=plt.subplot2grid((param_mean.size,param_mean.size), (j,i))
               for index in range(len(Fisher_list)):
                   param_cov = scipy.linalg.inv(Fisher_list[index])
                   twod_mean=np.array([param_mean[i],param_mean[j]])
                   twod_label=np.array([param_label[i],param_label[j]])
                   if len(Fisher_list[index]) == param_mean.size:
                       twod_cov_matrix = np.array([[param_cov[i,i],param_cov[i,j]], [param_cov[j,i],param_cov[j,j]]])
                       if colors is None:
                           plot_multiple_ellipse_sub(twod_mean, twod_cov_matrix, twod_label, figsub,index)
                       else:
                           plot_multiple_ellipse_sub(twod_mean, twod_cov_matrix, twod_label, figsub,index,color=colors[index])
                   elif len(Fisher_list[index]) != param_mean.size:
                       if i<len(Fisher_list[index]) and j<len(Fisher_list[index]):
                          twod_cov_matrix = np.array([[param_cov[i,i],param_cov[i,j]], [param_cov[j,i],param_cov[j,j]]])
                          if colors is None:
                              plot_multiple_ellipse_sub(twod_mean, twod_cov_matrix, twod_label, figsub,index)
                          else:
                              plot_multiple_ellipse_sub(twod_mean, twod_cov_matrix, twod_label, figsub,index,color=colors[index])

               if j != param_mean.size-1 :
                  figsub.set_xticklabels([])
                  figsub.yaxis.set_major_locator(plt.MaxNLocator(2))
               if i !=0  :
                  figsub.set_yticklabels([])
                  figsub.set_yticks([])
                  figsub.set_ylabel(str( ))
                  figsub.yaxis.set_major_locator(plt.MaxNLocator(2))
               else:
                   figsub.set_ylabel(param_label[j],fontsize=25)
                   figsub.yaxis.set_major_locator(plt.MaxNLocator(2))
                #Uncomment for limits on axes
               '''
               if i == 0:
                   figsub.set_xlim(param_mean[i]- 0.005,param_mean[i]+0.005)
               elif i == 1:
                   figsub.set_xlim(param_mean[i]- 0.06,param_mean[i]+0.06)
               elif i == 2:
                   figsub.set_xlim(param_mean[i]- 0.25,param_mean[i]+0.25)
               elif i == 3:
				   figsub.set_xlim(param_mean[i]- 0.5,param_mean[i]+0.5)
               else:
				   figsub.set_xlim(param_mean[i]- 0.4,param_mean[i]+0.4)

               if j == param_mean.size-1:
                   figsub.set_ylim(param_mean[j]-0.005,param_mean[j]+0.005)
               else:
                   figsub.set_ylim(param_mean[j]-0.05,param_mean[j]+0.05)
               '''
		#______________________________________________________________

               figsub.tick_params(axis='both', which='major', labelsize=18) #Default = 15
               figsub.set_xlabel(param_label[i],fontsize=25)
               figsub.xaxis.set_major_locator(plt.MaxNLocator(5))
               figsub.locator_params(axis="x", nbins=2)

           elif i==j:
               figsub=plt.subplot2grid((param_mean.size,param_mean.size), (j,i))
               for index in range(len(Fisher_list)):
                   param_cov = scipy.linalg.inv(Fisher_list[index])
                   if param_mean.size==len(Fisher_list[index]):
                       sig = np.sqrt(np.abs(param_cov[i,i]))
                       xx = np.linspace(param_mean[i]-2.*sig, param_mean[i]+2.*sig, 4000)
                       yy = 1./np.sqrt(2.*np.pi*sig**2.) * np.exp(-0.5 * ((xx-param_mean[i])/sig)**2.)
                       yy /= np.max(yy)
                       if colors is None:
                           figsub.plot(xx, yy, ls='solid', color='C'+str(index), lw=1.5, label=list_names[index]) #colours[k][0]
                           plt.ticklabel_format(style='plain')
                       else:
                           figsub.plot(xx, yy, ls='solid', color=colors[index], lw=1.5, label=list_names[index])
                           #figsub.ticklabel_format(style='plain')
                   elif param_mean.size!=len(Fisher_list[index]):
                       if i<len(Fisher_list):
                          sig = np.sqrt(np.abs(param_cov[i,i]))
                          xx = np.linspace(param_mean[i]-2.*sig, param_mean[i]+2.*sig, 4000)
                          yy = 1./np.sqrt(2.*np.pi*sig**2.) * np.exp(-0.5 * ((xx-param_mean[i])/sig)**2.)
                          yy /= np.max(yy)
                          if colors is None:
                              figsub.plot(xx, yy, ls='solid', color='C'+str(index), lw=1.5, label=list_names[index]) #colours[k][0]
                              #plt.ticklabel_format(style='plain')
                          else:
                              figsub.plot(xx, yy, ls='solid', color=colors[index], lw=1.5, label=list_names[index])
                              #plt.ticklabel_format(style='plain')

                   figsub.set_xlabel(param_label[i],fontsize=25)
                   figsub.tick_params(axis='both', which='major', labelsize=18) #Default =22
                   figsub.xaxis.set_major_locator(plt.MaxNLocator(2))
                   plt.ticklabel_format(style='plain')
                   handles, labels1 = figsub.get_legend_handles_labels() ##THIS IS THE STANDARD
                   #handles = [mpatches.Patch(color='orange', label='256 Dishes'),
                    #            mpatches.Patch(color='red', label='512 Dishes'),
                    #             mpatches.Patch(color='b', label='1024 Dishes') ]
                   #handles = [mpatches.Patch(color='red', label='HIRAX 1024'),
                    #             mpatches.Patch(color='b', label='HIRAX 1024 with Planck Prior')]
               if i is 0:
                   ###FOR LEGEND
                   fig.legend(handles,labels1,loc='upper right',fontsize=20)
                   #fig.legend(handles=handles,ncol=len(list_names),fontsize=35)#
                   #pass
               if i != param_mean.size-1:
                  figsub.set_xticklabels([])
                  #plt.ticklabel_format(style='plain')
               figsub.set_yticks([])


    fig.subplots_adjust(wspace=0, hspace=0)
    #fig.tight_layout()
    #if SAVEFIG: plt.savefig(output_dir + lbl + '_z_'+np.str(zA)+'_Multiple_Fisher_matrix.png')
    plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Main
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__=='__main__':


    zbinning_struct = binning.setup_ZBinningStruct(expt)
    if len(sys.argv)==2:
        zbin_index= int(sys.argv[1])
    else:
        zbin_index =2# int(sys.argv[1])



    zbin_prop = binning.setup_ZBinProp(zbin_index, zbinning_struct, expt)
    print('redshift center', zbin_prop.z_A)



    compute_SNR_Bl_21_21_kappa_cross(zbin_prop); raise KeyboardInterrupt
    #compute_SNR_Cl_21_kappa_cross(zbin_prop);raise KeyboardInterrupt
    #compute_SNR_Cl_21_21_auto(zbin_prop);raise KeyboardInterrupt
