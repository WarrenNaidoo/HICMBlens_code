######################################################################################################################
# This script computes the main results shown using the Full21cmblens_Fisher_matrices.py module to obtain the
# Fisher matrices for the w0waCDM case and used the Full21cmblens_primary_functions.py module to plot
# the results.
# The binning set up and experiment are chosen/set up in the binning.py and settings.py files respectively.
#####################################################################################################################


import Full21cmblens_Fisher_matrices as bl
import Full21cmblens_primary_functions as Fpm
import binning
import numpy as np
import settings
import HI_experiments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy
import scipy.linalg as la
expt=HI_experiments.getHIExptObject(settings.expt_name, settings.expt_mode)
from scipy.linalg import toeplitz
save_data_files = False


zbinning_struct = binning.setup_ZBinningStruct(expt)

cosmo_fisher21=np.zeros(169).reshape(13,13)
cosmo_fisher_kappa=np.zeros(25).reshape(5,5)
cosmo_fisher_bispec=np.zeros(169).reshape(13,13)
Planck_F = np.loadtxt('Planck_detf_fish.dat')

basic_fisher=np.array([])
full_fisher = np.array([])
z_list = np.array([])

#print('Planck prior', Planck_F)
#raise KeyBoardInterrupt

def add_fisher_matrices(F1, F2, lbls1, lbls2, info=False, expand=False):
    """
    Add two Fisher matrices that may not be aligned.
    """
    # If 'expand', expand the final matrix to incorporate the union of all parameters
    if expand:
        # Find params in F2 that are missing from F1
        not_found = [l for l in lbls2 if l not in lbls1]
        lbls = lbls1 + not_found
        Nnew = len(lbls)
        if info: print "add_fisher_matrices: Expanded output matrix to include non-overlapping params:", not_found

        # Construct expanded F1, with additional params at the end
        Fnew = np.zeros((Nnew, Nnew))
        Fnew[:F1.shape[0],:F1.shape[0]] = F1
        lbls1 = lbls
    else:
        # New Fisher matric is found by adding to a copy of F1
        Fnew = F1.copy()

    # Go through lists of params, adding matrices where they overlap
    for ii in range(len(lbls2)):
      if lbls2[ii] in lbls1:
        for jj in range(len(lbls2)):
          if lbls2[jj] in lbls1:
            _i = lbls1.index(lbls2[ii])
            _j = lbls1.index(lbls2[jj])
            Fnew[_i,_j] += F2[ii,jj]
            if info: print lbls1[_i], lbls2[ii], "//", lbls1[_j], lbls2[jj]
      if (lbls2[ii] not in lbls1) and info:
        print "add_fisher_matrices:", lbls2[ii], "not found in Fisher matrix."

    # Return either new Fisher matrix, or new (expanded) matrix + new labels
    if expand:
        return Fnew, lbls1
    else:
        return Fnew

def FOM(sig_x_sq, sig_y_sq , sig_x_y):
    #sig_x_sq, sig_y_sq , sig_x_y = np.float(sig_x_sq), np.float(sig_y_sq ), np.float( sig_x_y )
    alpha_sq = 6.17#1.52**2.
    det = sig_x_sq*sig_y_sq - sig_x_y**2.
    a_sq = (sig_x_sq + sig_y_sq)/2. + 0.25*np.sqrt( ((sig_x_sq - sig_y_sq)**2.) + (sig_x_y)**2.)
    b_sq = (sig_x_sq + sig_y_sq)/2. - 0.25*np.sqrt( ((sig_x_sq - sig_y_sq)**2.) + (sig_x_y)**2.)
    FOM = 1./(alpha_sq*np.sqrt(a_sq)*np.sqrt(b_sq))
    #return FOM
    #return 1./(sig_x_sq*sig_y_sq)
    return 0.25*np.pi/np.sqrt(det)
""" Uncomment to plot Planck Priors """

fid_val=np.array([0.962,0.022,0.67,0.67,0.836])
fid_val2=np.array([0.022,0.67,0.962,0.836,0.67])
lbl2 =np.array(['n_s',  'omega_b', 'omegaDE', 'h', 'sigma8'])
Planck_new_cov = np.loadtxt('Planck_2019.txt')

Planck_new_cov[:,4] = Planck_new_cov[:,4]/100
Planck_new_cov[4,:] = Planck_new_cov[4,:]/100
Planck_new = np.linalg.inv(Planck_new_cov)



Fixed = np.delete(np.delete(Planck_F,[1,2,4],axis=0),[1,2,4],axis=1)
F2 = np.delete(np.delete(Planck_F,[1,2,4],axis=0),[1,2,4],axis=1)

#######################################################################################################

''' COSMOLOGICAL CONSTRAINTS '''

# Compute Fisher matrices over the 4 bins
for i in range(0,4):
    zbin_index = i
    zbin_prop = binning.setup_ZBinProp(zbin_index, zbinning_struct, expt)
    z_list = np.append(z_list,zbin_prop.z_A)
    print('Redshift', zbin_prop.z_A)

    fid_label, fid_values, fisher_3params_21 = bl.get_three_param_FisherMatrix_Cl_21_21_auto_w0waCDM(zbin_prop)
    fid_label_kappa, fid_values_kappa, fisher_kappa = bl.get_Clkappa_distance_params_fisher_w0waCDM(zbin_prop)
    fid_label_bispec, fid_values_bispec, fisher_bispec = bl.get_distance_params_bispec_fisher_w0waCDM(zbin_prop)
    # Abao, sig8, b1, b2, f, aperp, apar
    # kappa sig8, DE, w0, wa, h


    derivs = bl. EOS_parameters_derivs()

    names_params, fid_val, fisher_6params_21 = bl.expand_fisher_matrix(zbin_prop, derivs, fisher_3params_21, fid_label)
    names_params_bispec, fid_val_bispec, fisher_6params_bispec = bl.expand_fisher_matrix(zbin_prop,derivs, fisher_bispec, fid_label_bispec)
    # Abao, sig8, b1, b2, f, aperp, apar, Om_k, Om_DE, w0, wa, h, gamma

    cosmo_fisher21 += fisher_6params_21
    if i == 0:

        cosmo_fisher_kappa =  fisher_kappa
    cosmo_fisher_bispec += fisher_6params_bispec
    # HI and Bispectrum here are Abao, sig8, b1, b2, f, aperp, apar, Om_k, Om_DE, w0, wa, h, gamma
    # Kappa is sigma_8, Om_DE, w0, wa, h



Planck_F = Planck_new
#lbl1 = ['omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
lbl1 = [ 'omegaDE', 'w0', 'wa', 'h']
#lbl2 =['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
lbl2 = ['omega_b','omegaDE','n_s',  'sigma8' , 'h']

# First fix HI Matrix over Abao b1 b2, f, aperp, apar, Om_k, gamma
cosmo_fisher21 = np.delete(np.delete(cosmo_fisher21,[0, 1, 2, 3, 4, 5,6,7,12],axis=1),[0, 1, 2, 3, 4, 5,6,7,12],axis=0)
# HI here are sig8  Om_DE, w0, wa, h



# Fix bispectrum over Abao, b1, b2, f,aperp, apar,Om_k, gamma
cosmo_fisher_bispec = np.delete(np.delete(cosmo_fisher_bispec, [0,1, 2,3,4,5,6,7,12], axis=1 ),[0,1,2,3,4,5,6,7,12],axis=0)
# Bispectrum here are sig8 Om_DE, w0, wa, h


# marginalise sigma_8 in kappa
cosmo_fisher_kappa_inv = np.linalg.inv(cosmo_fisher_kappa)
cosmo_fisher_kappa_margin =  np.delete(np.delete(cosmo_fisher_kappa_inv,[0],axis=1),[0],axis=0) # 'omegaDE', 'w0', 'wa', h
cosmo_fisher_kappa = np.linalg.inv(cosmo_fisher_kappa_margin ) # 'omegaDE', 'w0', 'wa', h


cosmo_sum = cosmo_fisher21 + cosmo_fisher_bispec


cosmo_comb = cosmo_fisher21  + cosmo_fisher_bispec + cosmo_fisher_kappa


Plank_only = np.zeros(25).reshape(5,5)

# Add Planck Priors
Fpl_21= add_fisher_matrices(cosmo_fisher21, Planck_F, lbl1, lbl2, expand=False)
Fpl_kappa= add_fisher_matrices(cosmo_fisher_kappa, Planck_F, lbl1, lbl2, expand=False)
Fpl_bispec = add_fisher_matrices( cosmo_fisher_bispec, Planck_F, lbl1, lbl2, expand=False)
Fpl_bispec_new = Fpl_bispec
Fpl_21_plus_bispec = add_fisher_matrices(cosmo_sum , Planck_F, lbl1, lbl2, expand=False)
Fpl_comb = add_fisher_matrices(cosmo_comb, Planck_F, lbl1, lbl2, expand=False) #
Pl_only = add_fisher_matrices(Plank_only, Planck_F, lbl1, lbl2, expand=False)


Prior_list = [Fpl_21, Fpl_bispec, Fpl_21_plus_bispec, Fpl_comb]


names_params = ['$\Omega_L$','$w0$','$wa$','$h$']



Prior_names = ['21cm Auto with Planck Prior', 'Bispectrum with Planck Prior',
                    '21cm Auto + Bispectrum with Planck Prior',
                    '21cm Auto + Bispectrum \n + Kappa Auto with Planck Prior']

fid_val = np.delete(fid_val,[0,5])

# Make contour plot
Fpm.plot_multiple_cov_ellipses('Cosmological Constraints with Prior' , 1.2, fid_val, Prior_list , Prior_names , names_params)


# Print covariance errors of HIRAX + Planck and Combined + Planck
print('HIRAX + Planck,  Combined + Planck Om_L',np.sqrt(np.linalg.inv(Fpl_21)[0,0]), np.sqrt(np.linalg.inv(Fpl_comb)[0,0]))
print('HIRAX + Planck,  Combined + Planck w0',np.sqrt(np.linalg.inv(Fpl_21)[1,1]), np.sqrt(np.linalg.inv(Fpl_comb)[1,1]))
print('HIRAX + Planck,  Combined + Planck wa',np.sqrt(np.linalg.inv(Fpl_21)[2,2]), np.sqrt(np.linalg.inv(Fpl_comb)[2,2]))
print('HIRAX + Planck,  Combined + Planck h',np.sqrt(np.linalg.inv(Fpl_21)[3,3]), np.sqrt(np.linalg.inv(Fpl_comb)[3,3]))
print('Advact errors',np.sqrt(np.diag(np.linalg.inv(Fpl_kappa))))


# Make w0-wa parameter plot with FoM in lagend
w0_cov = np.linalg.inv(Fpl_21)[1,1]
wa_cov = np.linalg.inv(Fpl_21)[2,2]
print('Figure of Merit HI', 1/np.sqrt(w0_cov*wa_cov- np.linalg.inv(Fpl_21)[1,2]**2. ))


axs1 =  plt.subplot(111)
param_cov0 = scipy.linalg.inv(Fpl_21)
param_cov1 = scipy.linalg.inv(Fpl_bispec)
param_cov2 = scipy.linalg.inv(Fpl_21_plus_bispec)
param_cov3 = scipy.linalg.inv( Fpl_comb )
param_cov_detf_plank = scipy.linalg.inv(Planck_F)
print('Planck_F',param_cov_detf_plank )

twod_mean=np.array([-1,0])
twod_label=np.array(['$w_0$','$w_a$'])

print('HI param',param_cov0[1,1])
twod_cov_matrix0 = np.array([[param_cov0[1,1],param_cov0[2,1]], [param_cov0[1,2],param_cov0[2,2]]])
twod_cov_matrix1 = np.array([[param_cov1[1,1],param_cov1[2,1]], [param_cov1[1,2],param_cov1[2,2]]])
twod_cov_matrix2 = np.array([[param_cov2[1,1],param_cov2[2,1]], [param_cov2[1,2],param_cov2[2,2]]])
twod_cov_matrix3 = np.array([[param_cov3[1,1],param_cov3[2,1]], [param_cov3[1,2],param_cov3[2,2]]])
twod_cov_matrix_planck = np.array([[param_cov_detf_plank[1,2],param_cov_detf_plank[1,2]], [param_cov_detf_plank[2,1],param_cov_detf_plank[2,2]]])


Fpm.plot_multiple_ellipse_sub(twod_mean, twod_cov_matrix0, twod_label, axs1, 0)
Fpm.plot_multiple_ellipse_sub(twod_mean, twod_cov_matrix1, twod_label, axs1, 1)
Fpm.plot_multiple_ellipse_sub(twod_mean, twod_cov_matrix2, twod_label, axs1, 2)
Fpm.plot_multiple_ellipse_sub(twod_mean, twod_cov_matrix3, twod_label, axs1, 3)



HI_FOM = (1/np.sqrt(twod_cov_matrix0[0][0]*twod_cov_matrix0[-1][-1]- twod_cov_matrix0[0][1]**2. ))/np.sqrt(np.pi*2)
print('HI FoM',HI_FOM)
label_HI = 'HI Auto, FoM:' + str(int(round(HI_FOM,0)))

B_FOM = (1/np.sqrt(twod_cov_matrix1[0][0]*twod_cov_matrix1[-1][-1]- twod_cov_matrix1[0][1]**2. ))/np.sqrt(np.pi*2)
label_B = 'Bispectrum, FoM:' + str(int(round(B_FOM,0)))


Sum_FOM = (1/np.sqrt(twod_cov_matrix2[0][0]*twod_cov_matrix2[-1][-1]- twod_cov_matrix2[0][1]**2. ))/np.sqrt(np.pi*2)
label_sum = 'HI Auto + Bispectrum, FoM:' + str(int(round(Sum_FOM,0)))


Comb_FOM = (1/np.sqrt(twod_cov_matrix3[0][0]*twod_cov_matrix3[-1][-1]- twod_cov_matrix3[0][1]**2. ))/np.sqrt(np.pi*2)
label_comb = 'HI Auto + Bispectrum + $\kappa$ Auto, \n FoM:' + str(int(round(Comb_FOM,0)))

leg0 = mpatches.Patch(color='C0', label=label_HI)
leg1 = mpatches.Patch(color='C1', label=label_B)
leg2 = mpatches.Patch(color='C2', label=label_sum)
leg3 = mpatches.Patch(color='C3', label=label_comb)

plt.legend(handles=[leg0, leg1,leg2,leg3], fontsize=22,loc='upper right')

axs1.tick_params(axis='both', which='major', labelsize=25)
plt.title('Flat $w_0 w_a$CDM',fontsize=25) #
plt.tick_params(axis='both',labelsize=25)
plt.xlabel(r'$w_0$', fontsize=30)
plt.ylabel(r'$w_a$', fontsize=30)
plt.tight_layout()
plt.show()
