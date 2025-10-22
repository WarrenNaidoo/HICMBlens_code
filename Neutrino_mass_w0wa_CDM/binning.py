import numpy as np
import matplotlib.pyplot as plt
import cosmolopy.distance as cd
from scipy.integrate import quad
from scipy import integrate

# Kavi
import HI_experiments
import settings

c=3e8 #speed of light in m/s
C_Mpc=3e5*(3.24078e-20) # Speed of light Mpc/s
import cosmological_parameters
cosmo = {'omega_M_0':cosmological_parameters.om_m, 'omega_lambda_0':cosmological_parameters.om_L, 'omega_k_0':0.0, 'h':cosmological_parameters.h}

# Z BINNING

class ZBinProp:
    def __init__(self, zbin_index, zbinning_struct, expt):

      self.deltanutilde = zbinning_struct.deltanutilde_channel  # Inherit frequency channel width in bin from global channel width (assume fixed for given expt)

      width_nutilde_arr, central_z_arr = zbinning_struct.width_nutilde_arr, zbinning_struct.central_z_arr

      self.Deltanutilde = width_nutilde_arr[zbin_index]
      self.z_A = central_z_arr[zbin_index]


      self.nutilde_A = 1.0/(1.0+self.z_A)
      self.chi_A = cd.comoving_distance(self.z_A, **cosmo)

      self.lambda_A = 0.21/self.nutilde_A # in metres
      self.FOV_A_sr = (self.lambda_A/expt.Ddish)**2
      self.theta_min_sr = (self.lambda_A/expt.Dmax)**2
      self.delta_chiperp_bin = np.sqrt(self.FOV_A_sr) * self.chi_A

      self.nu_tilde_min = self.nutilde_A - 0.5 * self.Deltanutilde
      self.nu_tilde_max = self.nutilde_A + 0.5 * self.Deltanutilde
      print('nu_min, nu_max', self.nu_tilde_min*1420, self.nu_tilde_max*1420)
      self.z_A_min=1.0/self.nu_tilde_max - 1.0
      self.z_A_max=1.0/self.nu_tilde_min - 1.0
      self.delta_z = self.z_A_max - self.z_A_min
      self.chi_A_min = cd.comoving_distance(self.z_A_min,**cosmo)
      self.chi_A_max = cd.comoving_distance(self.z_A_max,**cosmo)
      self.delta_chipar_bin = self.chi_A_max - self.chi_A_min

      self.rnu_A=self.get_rnu_A()


      self.y_min_expt = 0.01*self.rnu_A#settings.mode_factor_y/self.Deltanutilde #(Taking car of foreground cut)
      self.y_max_expt = settings.mode_factor_y/self.deltanutilde # ala Bull et al
      self.kpar_min = self.y_min_expt/self.rnu_A

      self.ell_min_expt = settings.mode_factor_ell/np.sqrt(self.FOV_A_sr)  # for interf use FOV for ell_min
      self.ell_max_expt = settings.mode_factor_ell/np.sqrt(self.theta_min_sr)
      self.kperp_min = self.ell_min_expt/self.chi_A
      self.knl = settings.knl_z0 * (1.0+self.z_A)**(2.0/(2.0+cosmological_parameters.ns))  	# Mpc^-1 units

      self.kfgnd = settings.kfgnd_z1

      self.qpar_min = 2.0*np.pi / self.delta_chipar_bin ; self.qpar_max = self.qpar_min * self.Deltanutilde/self.deltanutilde
      self.qperp_min = 2.0*np.pi / self.delta_chiperp_bin ; self.qperp_max = self.qperp_min * np.sqrt(self.FOV_A_sr/self.theta_min_sr)


      fsky = expt.getfsky()
      Sarea_sr = 4.*np.pi*fsky

      self.N_patches = Sarea_sr/self.FOV_A_sr
      self.Mode_Volume_Factor = self.Deltanutilde * Sarea_sr  				# U_bin = FOV_A_sr * Deltanutilde  ; Mode_Vol_Factor = U_bin * N_patches

    def get_ell_arr_expt(self):
	    delta_ell = settings.delta_ell_expt
	    n_ell_expt = 1 + np.int((self.ell_max_expt - self.ell_min_expt)/delta_ell)
	    ell_arr_expt=np.linspace(self.ell_min_expt,self.ell_max_expt,n_ell_expt)
	    return (n_ell_expt, ell_arr_expt)

    def	get_y_arr_expt(self):
        delta_y = settings.delta_y_expt
        n_y_expt = 1 + np.int((self.y_max_expt - self.y_min_expt)/delta_y)

        y_arr_expt=np.linspace(self.y_min_expt,self.y_max_expt,n_y_expt)
        return (n_y_expt, y_arr_expt)

    def get_kpar_arr_expt(self):
        y_arr = self.get_y_arr_expt()
        kpar = y_arr/self.rnu_A
        return kpar

    def get_kperp_arr_expt(self):
        ell_arr = self.get_ell_arr_expt()
        kperp = ell_arr/self.chi_A
        return kperp

    def get_rnu_A(self):
        delta_z=self.delta_z
        #z=self.z_A
        z=1.2


        zspace = np.linspace( z-delta_z, z+delta_z , 100 )
        int_1 = integrate.simps(C_Mpc/cd.hubble_z(zspace,**cosmo) , zspace)

        ztot = np.linspace(0,z,100)
        int_2 = integrate.simps(C_Mpc/cd.hubble_z(ztot,**cosmo), ztot)


        Ci = (1+z-delta_z)*(1+z+delta_z)*int_1/(2*delta_z*(1+z)*int_2)

        return self.chi_A*Ci*(1+z)

class ZBinningStruct:
    def __init__(self, expt):
        self.Dnutilde = (expt.numax/1.e6 - expt.numin/1.e6)/settings.nu21 # Full bandwidth in MHz
        self.num_channels = np.int((expt.numax/1.e6 - expt.numin/1.e6)/(expt.deltanu/1.e6)) # ~ 1000
        self.deltanutilde_channel = self.Dnutilde/self.num_channels # channel width

        self.num_z_bins = settings.num_z_bins

        if settings.Z_BIN_FORMAT == 'POWERLAW':
            self.min_nutilde_arr = expt.numin/1.e6/settings.nu21 + self.Dnutilde - \
    			(np.power(np.power(self.Dnutilde,settings.z_power_index)*np.linspace(1,self.num_z_bins,self.num_z_bins)/self.num_z_bins,1./settings.z_power_index))[::-1]
            self.width_nutilde_arr = np.append(np.diff(self.min_nutilde_arr),expt.numax/1.e6/settings.nu21-self.min_nutilde_arr[self.num_z_bins-1])
            self.max_nutilde_arr = self.min_nutilde_arr + self.width_nutilde_arr
            self.central_nutilde_arr = 0.5*(self.min_nutilde_arr + self.max_nutilde_arr)
            self.central_z_arr = 1.0/self.central_nutilde_arr - 1.0


        if settings.Z_BIN_FORMAT == 'LINEAR_100MHz_BINS':
            self.nu_bin_width = 100 #MHz
            self.min_nutilde_arr = expt.numin/(1.e6*settings.nu21) + self.nu_bin_width/settings.nu21 * np.arange(8)
            self.width_nutilde_arr = self.nu_bin_width/settings.nu21 * np.ones(4)
            self.max_nutilde_arr = self.min_nutilde_arr + self.width_nutilde_arr
            self.central_nutilde_arr = 0.5*(self.min_nutilde_arr + self.max_nutilde_arr)
            self.central_z_arr = 1.0/self.central_nutilde_arr - 1.0 ; print('z_i',self.central_z_arr)





def setup_ZBinningStruct(expt):
	ZBinning_Struct=ZBinningStruct(expt)
	return ZBinning_Struct


def setup_ZBinProp(zbin_index, zbinning_struct, expt):
	ZBin_Prop=ZBinProp(zbin_index, zbinning_struct, expt)
	return ZBin_Prop
