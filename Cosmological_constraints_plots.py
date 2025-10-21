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

cosmo_fisher21=np.zeros(169).reshape(13,13)
cosmo_fisher_kappa=np.zeros(225).reshape(15,15)
cosmo_fisher_kappa_F1=np.zeros(9).reshape(3,3)
cosmo_fisher_bispec=np.zeros(169).reshape(13,13)
#Planck_F = np.loadtxt('Planck_detf_fish.dat')

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



''' COSMOLOGICAL CONSTRAINTS '''

for i in range(0,4):
    zbin_index = i
    zbin_prop = binning.setup_ZBinProp(zbin_index, zbinning_struct, expt)
    z_list = np.append(z_list,zbin_prop.z_A)
    print('Redshift', zbin_prop.z_A)

    fid_label, fid_values, fisher_3params_21 = bl.get_three_param_FisherMatrix_Cl_21_21_auto(zbin_prop)
    fid_label_kappa, fid_values_kappa, fisher_kappa = bl.get_Clkappa_distance_params_fisher(zbin_prop)
    fid_label_bispec, fid_values_bispec, fisher_bispec = bl.get_distance_params_bispec_fisher(zbin_prop)

    # Matrices here are Abao, sig8, b1, ns, f, aperp, apar



    derivs = bl. EOS_parameters_derivs()

    names_params, fid_val, fisher_6params_21 = bl.expand_fisher_matrix(zbin_prop, derivs, fisher_3params_21, fid_label)
    names_params_bispec, fid_val_bispec, fisher_6params_bispec = bl.expand_fisher_matrix(zbin_prop,derivs, fisher_bispec, fid_label_bispec)
    # Matrices here are Abao, sig8, b1, ns, f, aperp, apar, Om_k, Om_DE, w0, wa, h, gamma
    # Kappa Matrices here are sig8, ns, Om_m, h



    cosmo_fisher21 += fisher_6params_21#[7:13,7:13]
    if i == 0:
        cosmo_fisher_kappa = fisher_kappa#[7:13,7:13]
    cosmo_fisher_bispec += fisher_6params_bispec #[7:13,7:13]





Planck_F = Planck_new
HI_fixed = np.delete(np.delete(cosmo_fisher21,[7,9,10,12],axis=1),[7,9,10,12],axis=0) # 'Abao, sig8, b1, ns, f, aperp, apar, Om_DE, h'
#kappa_fixed = np.delete(np.delete(cosmo_fisher_kappa,[0,2,6,8,9,11,12,14],axis=1),[0,2,6,8,9,11,12,14],axis=0)  #sig8, ns, Om_m, h,aperp, Om_DE, h'
Bispec_fixed = np.delete(np.delete(cosmo_fisher_bispec,[7,9,10,12],axis=1),[7,9,10,12],axis=0) #'' '''Abao, sig8, b1, ns, f, aperp, apar, Om_DE, h'
HI_cov = np.linalg.inv(HI_fixed)
Bispec_cov = np.linalg.inv(Bispec_fixed)



HI_marginalised = np.delete(np.delete(HI_cov,[0,2,4,5,6],axis=1),[0,2,4,5,6],axis=0) #'sig8, ns, Om_DE, h
Bispec_marginalised = np.delete(np.delete(Bispec_cov,[0,2,4,5,6],axis=1),[0,2,4,5,6],axis=0) #'sig8, ns, Om_DE, h


Bispec_cosmo_fish = np.linalg.inv(Bispec_marginalised)
HI_cosmo_fish = np.linalg.inv(HI_marginalised)
Kappa_cosmo_fish = cosmo_fisher_kappa # sig8, ns, Om_m, h




lbl1 = ['sigma8', 'n_s', 'omegaDE', 'h','omega_b']
lbl2 = ['omega_b','omegaDE','n_s',  'sigma8' , 'h']
lbl3 = ['sigma8', 'n_s', 'omegaDE', 'h','aperp']
lbl4 = ['h'] #'omegaDE',
lbl5 = ['sigma8', 'n_s', 'omegaDE', 'h']

cosmo_kappa_F1_blank = np.array(np.zeros(25).reshape(5,5))


fid_val = np.array([ 0.8159,0.9667,0.3089,0.678])
names_params = ['$\sigma_8$','$n_s$','$\Omega_M$','$h$']


Kappa_total = add_fisher_matrices(cosmo_kappa_F1_blank, Kappa_cosmo_fish,lbl1, lbl5, expand=False)

Plank_only = np.zeros(25).reshape(5,5)


HI_cosmo_fish = np.vstack([HI_cosmo_fish,np.zeros(4)])
HI_cosmo_fish = np.column_stack([HI_cosmo_fish,np.zeros(5)])


Bispec_cosmo_fish = np.vstack([Bispec_cosmo_fish,np.zeros(4)])
Bispec_cosmo_fish = np.column_stack([Bispec_cosmo_fish,np.zeros(5)])


#Kappa_cosmo_fish = np.vstack([Kappa_cosmo_fish,np.zeros(4)])
#Kappa_cosmo_fish = np.column_stack([Kappa_cosmo_fish,np.zeros(5)])

Kappa_total[:,1] = 0
Kappa_total[1,:] = 0


cosmo_sum = HI_cosmo_fish + Bispec_cosmo_fish
cosmo_comb = HI_cosmo_fish + Bispec_cosmo_fish + Kappa_total

Fpl_21= add_fisher_matrices(HI_cosmo_fish, Planck_F, lbl1, lbl2, expand=False)
Fpl_bispec = add_fisher_matrices( Bispec_cosmo_fish, Planck_F, lbl1, lbl2, expand=False)
Fpl_21_plus_bispec = add_fisher_matrices(cosmo_sum , Planck_F, lbl1, lbl2, expand=False)
Fpl_comb = add_fisher_matrices(cosmo_comb, Planck_F, lbl1, lbl2, expand=False) #
Pl_only = add_fisher_matrices(Plank_only, Planck_F, lbl1, lbl2, expand=False)
Fpl_kappa = add_fisher_matrices(Kappa_total, Planck_F, lbl1, lbl2, expand=False)


Prior_names = ['Planck','21cm Auto with Planck Prior', 'Kappa w prior',
                    '21cm Auto + Bispectrum \n + kappa with Planck Prior']
Prior_list = [Fpl_21[0:4,0:4], Fpl_kappa[0:4,0:4], Fpl_21_plus_bispec[0:4,0:4], Fpl_comb[0:4,0:4] ]



Prior_names = ['HI', 'kappa', 'HI + bispec', 'Comb all w priors']
fid_val = np.array([ 0.8159,0.9667,0.3089,0.678])
names_params = ['$\sigma_8$','$n_s$','$\Omega_M$','$h$']

bl.plot_multiple_cov_ellipses('Cosmological Constraints with Prior' , zbin_prop.z_A, fid_val, Prior_list , Prior_names , names_params )

print('HIRAX + Planck,  Combined + Planck sigma8',np.sqrt(np.linalg.inv(Fpl_21)[0,0]), np.sqrt(np.linalg.inv(Fpl_comb)[0,0]))
print('HIRAX + Planck,  Combined + Planck ns',np.sqrt(np.linalg.inv(Fpl_21)[1,1]), np.sqrt(np.linalg.inv(Fpl_comb)[1,1]))
print('HIRAX + Planck,  Combined + Planck Omega_M',np.sqrt(np.linalg.inv(Fpl_21)[2,2]), np.sqrt(np.linalg.inv(Fpl_comb)[2,2]))
print('HIRAX + Planck,  Combined + Planck h',np.sqrt(np.linalg.inv(Fpl_21)[3,3]), np.sqrt(np.linalg.inv(Fpl_comb)[3,3]))
print('HIRAX + Planck,  Combined + Planck Om_b',np.sqrt(np.linalg.inv(Fpl_21)[4,4]), np.sqrt(np.linalg.inv(Fpl_comb)[4,4]))


sig_8_cov = np.linalg.inv(Fpl_21)[0,0]
Om_m_cov = np.linalg.inv(Fpl_21)[2,2]
print('Figure of Merit HI', (1/np.sqrt(sig_8_cov*Om_m_cov - np.linalg.inv(Fpl_21)[0,2]**2. ))/np.sqrt(np.pi*2))

pnames =  names_params[7:13]
axs1 =  plt.subplot(111)
param_cov0 = scipy.linalg.inv(Fpl_21)
param_cov1 = scipy.linalg.inv(Fpl_bispec)
param_cov2 = scipy.linalg.inv(Fpl_21_plus_bispec)#scipy.linalg.inv(Fpl_21 + Fpl_bispec)
param_cov3 = scipy.linalg.inv( Fpl_comb )
param_cov_detf_plank = scipy.linalg.inv(Pl_only)
print('Planck_F',param_cov_detf_plank )

twod_mean=np.array([0.8159,0.3089])
twod_label=np.array(['$\sigma_8$','$\Omega_m$'])

twod_cov_matrix0 = np.array([[param_cov0[0,0],param_cov0[0,2]], [param_cov0[2,0],param_cov0[2,2]]])
twod_cov_matrix1 = np.array([[param_cov1[0,0],param_cov1[0,2]], [param_cov1[2,0],param_cov1[2,2]]])
twod_cov_matrix2 = np.array([[param_cov2[0,0],param_cov2[0,2]], [param_cov2[2,0],param_cov2[2,2]]])
twod_cov_matrix3 = np.array([[param_cov3[0,0],param_cov3[0,2]], [param_cov3[2,0],param_cov3[2,2]]])
twod_cov_matrix_planck = np.array([[param_cov_detf_plank[0,0],param_cov_detf_plank[0,2]], [param_cov_detf_plank[2,0],param_cov_detf_plank[2,2]]])




bl.plot_multiple_ellipse_sub(twod_mean, twod_cov_matrix0, twod_label, axs1, 0, color='C0')
bl.plot_multiple_ellipse_sub(twod_mean, twod_cov_matrix1, twod_label, axs1, 1, color='C1')
bl.plot_multiple_ellipse_sub(twod_mean, twod_cov_matrix2, twod_label, axs1, 2, color='C2')
bl.plot_multiple_ellipse_sub(twod_mean, twod_cov_matrix3, twod_label, axs1, 3, color='C3')



leg0 = mpatches.Patch(color='C0', label='HI Auto')
leg1 = mpatches.Patch(color='C1', label='Bispectrum')
leg2 = mpatches.Patch(color='C2', label='HI Auto + Bispectrum')
leg3 = mpatches.Patch(color='C3', label='HI Auto + Bispectrum + $\kappa$ Auto')


plt.legend(handles=[leg0, leg1,leg2,leg3], fontsize=15,loc='upper right')

axs1.tick_params(axis='both', which='major', labelsize=18)
#plt.title('Dark Energy Constraints',fontsize=18) #
plt.tick_params(axis='both',labelsize=18)
#plt.ylim([-0.3,0.3])
#plt.xlim([-1.15,-0.85])
plt.tight_layout()
plt.show()

