import Full21cmblens as bl
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

cosmo_fisher21=np.zeros(196).reshape(14,14)
cosmo_fisher_kappa=np.zeros(36).reshape(6,6)
cosmo_fisher_bispec=np.zeros(196).reshape(14,14)

basic_fisher=np.array([])
full_fisher = np.array([])
z_list = np.array([])


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

    return 0.25*np.pi/np.sqrt(det)


Planck_new_cov = np.loadtxt('Planck_2019.txt')
Planck_new_cov[:,4] = Planck_new_cov[:,4]/100
Planck_new_cov[4,:] = Planck_new_cov[4,:]/100
Planck_new = np.linalg.inv(Planck_new_cov)



#######################################################################################################

''' COSMOLOGICAL CONSTRAINTS '''

for i in range(0,4):
    zbin_index = i
    zbin_prop = binning.setup_ZBinProp(zbin_index, zbinning_struct, expt)
    z_list = np.append(z_list,zbin_prop.z_A)
    print('Redshift', zbin_prop.z_A)

    fid_label, fid_values, fisher_3params_21 = bl.get_three_param_FisherMatrix_Cl_21_21_auto(zbin_prop)
    fid_label_kappa, fid_values_kappa, fisher_kappa = bl.get_Clkappa_distance_params_fisher(zbin_prop)
    fid_label_bispec, fid_values_bispec, fisher_bispec = bl.get_distance_params_bispec_fisher(zbin_prop)
    # Matrices here have parameters 'Abao','sig8', 'b1', 'b2', 'Mv', 'f', 'aperp', 'apar'
    # Kappa is sigma_8, Mv, Om_DE, w0, wa, h



    derivs = bl. EOS_parameters_derivs()

    names_params, fid_val, fisher_6params_21 = bl.expand_fisher_matrix(zbin_prop, derivs, fisher_3params_21, fid_label)
    names_params_bispec, fid_val_bispec, fisher_6params_bispec = bl.expand_fisher_matrix(zbin_prop,derivs, fisher_bispec, fid_label_bispec)
    # Matrices here have parameters 'Abao','sig8', 'b1', 'b2', 'Mv', 'f', 'aperp', 'apar', Omk, OmDE, w0, wa, h, gamma

    if i == 0:
        cosmo_fisher_kappa =  fisher_kappa
    cosmo_fisher21 += fisher_6params_21#[8:14,8:14]
    cosmo_fisher_bispec += fisher_6params_bispec#[8:14,8:14]


# First fix b2 for HI and aperp, apar for bispectrum
# HI
cosmo_fisher21 = np.delete(np.delete(cosmo_fisher21, [3,6,7,8,13], axis=1), [3,6,7,8,13], axis=0)
# HI now: 'Abao','sig8', 'b1', 'Mv', 'f', OmDE, w0, wa, h
# Bispec
cosmo_fisher_bispec = np.delete(np.delete(cosmo_fisher_bispec, [6,7,8,13], axis=1), [6,7,8,13], axis=0)
# Bispec now: 'Abao','sig8', 'b1', 'b2' 'Mv', 'f', OmDE, w0, wa, h



cosmo_fisher21_inv = np.linalg.inv(cosmo_fisher21)
cosmo_fisher21_margin  = np.delete(np.delete(cosmo_fisher21_inv, [0,1,2,4], axis=1), [0,1,2,4], axis=0)
cosmo_fisher21 = np.linalg.inv(cosmo_fisher21_margin) # 'Mv', OmDE, w0, wa, h

cosmo_fisher_bispec_inv = np.linalg.inv(cosmo_fisher_bispec)
cosmo_fisher_bispec_margin = np.delete(np.delete(cosmo_fisher_bispec_inv, [0,1,2,3,5], axis=1), [0,1,2,3,5], axis=0)
cosmo_fisher_bispec = np.linalg.inv(cosmo_fisher_bispec_margin ) #'Mv', OmDE, w0, wa, h




Planck_F = Planck_new
lbl1 = [ 'M_v', 'omegaDE', 'w0', 'wa', 'h']
lbl2 = ['omega_b','omegaDE','n_s',  'sigma8' , 'h']


cosmo_fisher_kappa_inv = np.linalg.inv(cosmo_fisher_kappa)
cosmo_fisher_kappa_margin = np.delete(np.delete(cosmo_fisher_kappa_inv,[0],axis=1),[0],axis=0)
cosmo_fisher_kappa = np.linalg.inv(cosmo_fisher_kappa_margin)   #Mv, Om_DE, w0, wa, h

cosmo_sum = cosmo_fisher21 + cosmo_fisher_bispec


cosmo_comb = cosmo_fisher21  + cosmo_fisher_bispec + cosmo_fisher_kappa


Plank_only = np.zeros(25).reshape(5,5)


Fpl_21= add_fisher_matrices(cosmo_fisher21, Planck_F, lbl1, lbl2, expand=False)
Fpl_kappa= add_fisher_matrices(cosmo_fisher_kappa, Planck_F, lbl1, lbl2, expand=False)
Fpl_bispec = add_fisher_matrices( cosmo_fisher_bispec, Planck_F, lbl1, lbl2, expand=False)
Fpl_bispec_new = Fpl_bispec
Fpl_21_plus_bispec = add_fisher_matrices(cosmo_sum , Planck_F, lbl1, lbl2, expand=False)
Fpl_comb = add_fisher_matrices(cosmo_comb, Planck_F, lbl1, lbl2, expand=False) #
Pl_only = add_fisher_matrices(Plank_only, Planck_F, lbl1, lbl2, expand=False)


Prior_list = [Fpl_21, Fpl_kappa, Fpl_21_plus_bispec, Fpl_comb]


names_params = [r'$M_{\nu}$','$\Omega_M$','$w0$','$wa$','$h$']



Prior_names = ['21cm Auto with Planck Prior', '$\kappa$ auto with Planck Prior',
                    '21cm Auto + Bispectrum with Planck Prior',
                    '21cm Auto + Bispectrum \n + Kappa Auto with Planck Prior']

fid_val = np.delete(fid_val,[0,1, 5])
fid_val = np.insert(fid_val, [0,0], [0.003, 0.308])


bl.plot_multiple_cov_ellipses('Cosmological Constraints with Prior' , 1.2, fid_val, Prior_list , Prior_names , names_params)

Prior_names = ['HI Auto', '$\kappa$ Auto',
                    'HI Auto + Bispectrum',
                    'HI Auto + Bispectrum + Kappa Auto']

Prior_list = [cosmo_fisher21, cosmo_fisher_kappa, cosmo_sum, cosmo_comb]

bl.plot_multiple_cov_ellipses('Cosmological Constraints with Prior' , 1.2, fid_val, Prior_list , Prior_names , names_params)



# To fix w0 wa to see constraints on Neutrino mass, comment out for constraint on Nuetrino mass with varying w0 and wa.
Fpl_comb  = np.delete(np.delete(Fpl_comb, [2,3],axis=1 ), [2,3],axis=0)
