# python -m py_compile NEW.py for New Files

import cosmological_parameters

SAVEFIG = True # False 					# True or False
SHOWFIG = True                      # True or False

hubble_inverse = cosmological_parameters.h
hubble_inverse_volume = cosmological_parameters.h**3

expt_name='hirax'#'meerkat'#'hirax'
expt_mode='interferometer'
nu21 = 1420.405 # MHz

num_z_bins = 4
Z_BIN_FORMAT = 'POWERLAW'#'LINEAR_100MHz_BINS'#			'POWERLAW' # Option for equal 'SN' NOT yet setup
z_power_index = 0.55				# to give more SN to high-z bins decrease index - 1 is equal binning

pi = 3.14159

sq_deg_to_sr = (pi/180.)**2 		# Flat sky approx

mode_factor_y = 2.0 * pi
delta_y_expt = 10#5 					# Must be high enough resolution to capture main features in power spectra

mode_factor_ell = 2*pi
delta_ell_expt = 4#2

knl_z0 = 0.2 * cosmological_parameters.h      	# Mpc^-1
kfgnd_z1 = 0.01 * cosmological_parameters.h		# Mpc^-1
