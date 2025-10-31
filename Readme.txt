----------------------------------------
HICMBlens_code
--------------
Authors: Warren Naidoo, Kavilan Moodley, Heather Price and Aurelie Penin
Email: warrendoo855@gmail.com
October 2025
----------------------------------------

This code is used to obtain cosmological Fisher forecasts for HI-CMB lensing cross-bispectrum surveys.

About:
------
The HICMBlens_code computes Fisher forecasts for HI intensity mapping experiments
cross-correlated with the CMB lensing convergence field in a three-point
bispectrum estimator.  The formalism is this code is based on the results presented Moodley
et al. arXiv: 2311.05904. This code is in Python and uses many of Pythons in-built libraries
such as NumPy, SciPy, and matplotlib and other cosmology based packages like CAMB and cosmolopy.

Requirements
------------

 - Python 2.7
 - Recent NumPy and SciPy
 - matplotlib
 - CAMB
 - Cosmolopy
 
 Getting Started and using the code:
 -----------------------------------

The HICMBlens_code package consists of a collection of scripts that require no installation beyond the dependencies listed above. To begin, users can select a desired HI experiment in the settings.py file. The expt variable within settings.py is utilized by the HI_experiments.py script to compute the response and noise power spectrum for the specified HI experiment. While several predefined experiments are provided—primarily focusing on the HIRAX experiment for our analysis—users are free to define and include details of any other HI experiment to obtain the corresponding results.


Next, the redshift binning scheme can be configured in the settings.py file. The Z_BIN_FORMAT parameter specified in settings.py is utilized by the binning.py script to generate the desired bins over a specified redshift range. Two predefined binning schemes are currently provided: 
(1) a logarithmic (power-law) binning method that yields approximately uniform HI signal-to-noise across four bins, and 
(2) evenly spaced bins of 100 MHz width. 
Users may easily define alternative binning schemes—such as uniformly spaced redshift bins (e.g., delta_z = 0.1) by modifying the binning.py file accordingly.

For the CMB lensing experiment, the desired configuration can be specified within the CMB_lensing.py module using the getN0kappa(ls, expt, spec) function. The code currently includes predefined options for the AdvACT, Planck, and a more optimistic reference (“ref”) CMB lensing experiments. Users may also provide custom noise files corresponding to other CMB lensing experiments to generate results tailored to their specific configurations.

To reproduce the main results presented in arXiv:2311.05904, the Full21cmblens_primary_functions.py module should be executed as the main script to obtain the 2D signal-to-noise plots for both the HI auto-spectrum and bispectrum (user can toggle between which result they want to under the 'main' section of this module). This module also computes the HI power spectrum, CMB lensing power spectrum, and their cross-bispectrum. 

The variable zbin_index in the main script specifies the redshift bin for which the signal-to-noise is computed. In the “power law” binning configuration, the redshift centers are (0.81, 0.95, 1.27, 1.95), corresponding to zbin_index values of (3, 2, 1, 0), respectively.

To compute the Fisher matrix results, we note that firstly all Fisher derivatives and Fisher matrices are written in the 'Full21cmblens_Fisher_matrices.py' file. In the 'Full21cmblens_Fisher_matrices.py' one can obtain all Fisher matrices for the distance paremeters (i.e b, f, sigma_8 etc.) for each probe (HI auto, CMB lensing, bispectrum). Next, all Fisher derivatives and matrices for the three cosmological models - LCDM, w0waCDM, and Neutrino-mass-w0waCDM are also evaluated using the Full21cmblens_Fisher_matrices.py module. To reproduce the results presented in arXiv:2311.05904, run the scripts Cosmological_constraints_LCDM.py, Cosmological_constraints_w0waCDM.py, and Cosmological_constraints_Neutrino_mass.py. Each of these scripts calls the corresponding Fisher matrices from the Full21cmblens_Fisher_matrices.py module as well as the transformation matrix 'EOS_parameters_derivs()' used to expand the distance parameter matrices to one involving the cosmological paramters through the 'Alcock–Paczyński (AP) effect'. These scripts performs the computation across the four redshift bins. The resulting Fisher matrices are then processed, combined, and supplemented with Planck priors before being plotted using the visualization tools defined in the Full21cmblens_primary_functions.py module.
