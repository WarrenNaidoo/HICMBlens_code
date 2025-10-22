import numpy as np
import matplotlib.pyplot as plt
import cosmolopy.distance as cd
#from scipy.interpolate import InterpolatedUnivariateSpline as interp1d
from scipy.integrate import quad
from scipy.interpolate import interp1d
import scipy
import binning

import settings # Kavi

c=3e8 #speed of light in m/s
C_Mpc=3e5*(3.24078e-20)
import cosmological_parameters
cosmo = {'omega_M_0':cosmological_parameters.om_m, 'omega_lambda_0':cosmological_parameters.om_L, 'omega_k_0':0.0, 'h':cosmological_parameters.h}
use_Bull_n_x=False


class HIExperiment:
    def __init__(self, name, Tins, tobs, Sarea, Dmax, Ddish, Ndish, numin, numax, deltanu, mode, npol=2, Nbeams=1):
        self.name=name                 #experiment name
        self.Tins=float(Tins)          #instrument temp in Kelvin (add sky temp to get Tsys)
        self.tobs=float(tobs)          #observation time in seconds
        self.Sarea=float(Sarea)        #survey area in square degrees
        self.Dmax=float(Dmax)          #longest baseline in m
        self.Ddish=float(Ddish)        #dish diameter in m
        self.Ndish=float(Ndish)        #number of dishes
        self.numin=float(numin)        #minimum frequency in Hz
        self.numax=float(numax)        #maximum frequency in Hz
        self.deltanu=float(deltanu)    #bandwith in Hz
        self.mode=mode                 #Interfermeter or dish mode
        self.npol=npol                 #number of polarisation channels
        self.Nbeams=Nbeams             #number of beams - default 1

    def	getfsky(self):
	    Sarea_sr = self.Sarea*settings.sq_deg_to_sr
	    fsky = Sarea_sr/(4.*np.pi)
	    return fsky


    def getFcover(self):    #fraction of the area of the array covered with dishes
        Fcover = self.getAcoll()/(np.pi*(self.Dmax/2)**2)
        if Fcover>1:
            return 1
        return Fcover

    def getLmax(self, nu=None):  #maximum multipole measureable at frequency nu
        if nu is None:
            nu=(self.numax+self.numin)/2
        return 1500

    def getAcoll(self):
        return np.pi*(self.Ddish/2)**2*self.Ndish

    def getDishArea(self):
        return np.pi*(self.Ddish/2)**2


    def getThetaBeam(self, nu=None):
        if nu is None:
            nu=(self.numax+self.numin)/2
        return (3e8/nu)/self.Ddish

    def getNumDensityInUV(self, ell, nu):
        nu_MHz=nu/1e6
        print self.name
        if use_Bull_n_x:
            if self.name=='Meerkat':
                data=np.loadtxt('BaselineDataFiles/radiofisher_array_config/nx_MKREF2_dec90.dat')
                print 'Meerkat - using Bull n(x)'
            elif self.name=='SKA1':
                data=np.loadtxt('BaselineDataFiles/radiofisher_array_config/nx_SKAMREF2_dec90.dat')
                print 'SKA1 - using Bull n(x)'
            elif self.name=='CHIME':
                data=np.loadtxt('BaselineDataFiles/radiofisher_array_config/nx_CHIME_800.dat')
            else:
                print 'Sorry, no n_x for experiment by that name'
                return
            x=data[:,0]
            n_x=data[:,1]
            n_x_interp=interp1d(x, n_x,bounds_error=False, fill_value=0.0)
            x_l=ell/(2*np.pi*nu_MHz)
            n_x_l=n_x_interp(x_l)
            n_u=n_x_l/(nu_MHz)**2

        else:
            data=np.loadtxt('BaselineDataFiles/n_D_' +self.name+'.dat')
            D_bin=data[:,0]                 #physical baseline lengths - histogram bins
            n_D_bin=data[:,1]               #number density
            n_D_interp=interp1d(D_bin, n_D_bin, bounds_error=False,fill_value=0.0)

            wavelength=c/nu
            D=wavelength*ell/(2*np.pi)
            n_D_new=n_D_interp(D)
            n_u=n_D_new*wavelength**2

        return n_u

    def getNumDensityUniform(self, ell, nu):
        n=np.loadtxt('BaselineDataFiles/n_'+self.name+'_uniform.dat')
        wavelength=c/nu
        n_u=np.ones(ell.shape)*n*wavelength**2
        return n_u

    def getfnu(self, nu):   #see Bull et al appendix D
        return 1

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Main noise function
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def getNoiseAngularPower(self, ell, nu,zbin_prop, y=1.0, B=400e6):  #from Pourtsidou et al HI x Galaxy paper, Bull et al 2014
        #returns thermal noise array corresponding to ell array for a fixed frequency (is this the best way?)
        Tsky=getTsky(nu)    #K - convert to mK later
        Tsys=self.Tins+Tsky
        nu_21=float(1420e6)
        eff_factor = 0.5
        if nu is None:
            nu=(self.numax+self.numin)/2.
        wavelength=c/nu
        if self.name=='CHIME':
            fov=np.pi*wavelength/self.Ddish
        else:
            fov=wavelength**2/self.Ddish**2.

        #n=self.getNumDensityUniform(ell, nu)
        if self.mode[0].lower()=='i':   #interferometer
            n=self.getNumDensityInUV(ell, nu)
            nfact=np.array([])
            for jj in n:
                if jj==0:
                    nu=1e-20
                else:
                    nu=jj
                nfact = np.append(nfact,nu)
            int_dish_factor=fov/nfact
        elif self.mode[0].lower()=='d':
            beam_perp=np.exp(-1*(ell*self.getThetaBeam())**2/(16*np.log(2)))
            multiplicity_factor=self.getfnu(nu)/(self.Nbeams*self.Ndish)
            int_dish_factor=multiplicity_factor/beam_perp**2


        sq_deg_to_sr = (np.pi/180.)**2
        S=1e6
        C_N=Tsys**2*self.Sarea*sq_deg_to_sr/(self.npol*self.tobs*self.deltanu)*int_dish_factor*self.deltanu/nu_21 #Bull

        print 'Tsys:', Tsys
        print 'fov:',fov
        print 'wavelength:', wavelength
        print 'Adish:',self.getDishArea()

        return np.abs(C_N*eff_factor)*1e6  #convert from K^2 to mK^2

    def getPlottableNoise(self, ell, nu=None):  #Match Bull appendix plot
        Tsky=getTsky(nu)
        Tsys=self.Tins+Tsky
        nu_21=float(1420e6)
        if nu is None:
            nu=(self.numax+self.numin)/2.
        wavelength=c/nu
        fov=wavelength**2/self.getDishArea()

        if self.mode[0].lower()=='i':
            no=np.array([])
            n=self.getNumDensityInUV(ell, nu)

            for jj in n:
                if jj==0:
                    nu=1e-10
                else:
                    nu=jj
                no=np.append(no,nu)
            int_dish_factor=fov/no
        elif self.mode[0].lower()=='d':
            beam_perp=np.exp(-1*(ell*self.getThetaBeam(nu))**2/(16*np.log(2)))
            multiplicity_factor=self.getfnu(nu)/(self.Nbeams*self.Ndish)
            int_dish_factor=multiplicity_factor/beam_perp**2

        n=self.getNumDensityInUV(ell, nu)
        return Tsys**2*np.abs(int_dish_factor)

    #for P(k) noise:
    def getWsq(self,ell, nu):
        Wsq=np.exp(-ell**2*(self.getThetaBeam(nu)**2/(8*np.log(2))))
        return Wsq

    def getSigmaPix(self, nu):
        deltaf=50e3 #frequency resolution from Pourtsidou et al
        omega_pix=self.getThetaBeam(nu)**2*1.13 #Pourtsidou et al
        omega_tot=self.Sarea*(np.pi/180)**2
        sigma_p=self.Tins/np.sqrt(deltaf*self.tobs*omega_pix/omega_tot*self.Ndish)
        return sigma_p



    def getSurveyVolume(self, omega):
        zmax=1420e6/self.numin-1
        zmin=1420e6/self.numax-1

        I,err=quad(lambda z: c*cd.comoving_distance(z, **cosmo)**2/cd.hubble_z(z, **cosmo), zmin, zmax)
        print '-------------------\nin getSurveyVolume\n-------------------'
        print 'integral:',I
        print 'error:',err
        print 'solid angle:',omega
        print 'H(zmax):', cd.hubble_z(zmax, **cosmo)
        print 'H(zmin):', cd.hubble_z(zmin, **cosmo)
        print 'chi(zmax):', cd.comoving_distance(zmax, **cosmo)
        print 'chi(zmin):', cd.comoving_distance(zmin, **cosmo)
        print zmin
        print zmax

        V=omega*I
        return V




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def getVolumeIntegrand(z):
    integrand=c*cd.comoving_distance(z, **cosmo)**2/cd.hubble_z(z, **cosmo)


def getHIExptObject(n, mode='interferometer', tobsyears = 2.):
    print('Time obs', tobsyears)
    tobs=tobsyears*365.25*24*60*60

    global use_Bull_n_x
    if n.lower()=='hirax':
        expt=HIExperiment(name='HIRAX', Tins=50, tobs=tobs, Sarea=15000, Dmax=263, Ddish=6, Ndish=1024, numin=400e6, numax=800e6, deltanu=0.4e6, mode=mode)#1024 dishes so I can make a regular square grid for baselines, Sarea=10000 sq degrees for overlap with CMB lensing with fsky=0.25
        #expt=HIExperiment(name='HIRAX', Tins=50, tobs=tobs, Sarea=2000, Dmax=263, Ddish=6, Ndish=256, numin=400e6, numax=800e6, deltanu=0.4e6, mode=mode)
        #default parameters may be approximate, deltanu=20 corresponds to Pourtsidou et al (change?)
        #expt=HIExperiment(name='HIRAX', Tins=50, tobs=tobs, Sarea=15000, Dmax=260, Ddish=8, Ndish=256, numin=400e6, numax=800e6, deltanu=0.4e6, mode=mode)#576 dishes
        #expt=HIExperiment(name='HIRAX', Tins=50, tobs=tobs, Sarea=15000, Dmax=305, Ddish=8, Ndish=512, numin=400e6, numax=800e6, deltanu=0.4e6, mode=mode)#768 dishes
        #CHORD
        #expt=HIExperiment(name='HIRAX', Tins=30., tobs=tobs, Sarea=14000., Dmax=22*6., Ddish=6., Ndish=512, numin=400e6, numax=800e6, deltanu=0.4e6, mode=mode)#1024 dishes so I can make a regular square grid for baselines, Sarea=10000 sq degrees for overlap with CMB lensing with fsky=0.25

        use_Bull_n_x=False
        print 'HIRAX object initialised'
    elif n.lower()=='ska1':
        expt=HIExperiment(name='SKA1', Tins=28, tobs=tobs, Sarea=25000, Dmax=4000, Ddish=15, Ndish=190, numin=350e6, numax=1050e6, deltanu=20e6, mode=mode)  #check survey area for interferometric mode
        use_Bull_n_x=True
        print 'SKAmid1 object initialised with area calculated from specs listed in Bull et al 2014'
    elif n.lower()=='meerkat':
        expt=HIExperiment(name='Meerkat', Tins=29, tobs=tobs, Sarea=5000, Dmax=4000, Ddish=13.5, Ndish=64, numin=580e6, numax=1015e6, deltanu=20e6, mode=mode)    #Band 1 for comparison with HIRAX
        use_Bull_n_x=True
        print 'Meerkat band 1 object initialised with area calculated from specs listed in Bull et al 2014'
    elif n.lower()=='chime':
        expt=HIExperiment(name='CHIME', Tins=50, tobs=tobs, Sarea=25000, Dmax=300, Ddish=20, Ndish=1280, numin=400e6, numax=800e6, deltanu=20e6, mode=mode)
        use_Bull_n_x=True

    return expt

"""
def getHIRAXobject(name='HIRAX', Tins=50, tobsyears=2, Sarea=15000, Dmax=263, Ddish=6, Ndish=1024, numin=400e6, numax=800e6, deltanu=20e6): #1024 dishes so I can make a regular square for baselines
    #default parameters may be approximate, deltanu corresponds to Pourtsidou et al (change?)
    tobs=tobsyears*365.25*24*60*60
    HIRAX=HIExperiment(name, Tins, tobs, Sarea, Dmax, Ddish, Ndish, numin, numax, deltanu)
    print 'HIRAX object initialised'
    return HIRAX

def getSKAmid1object(name='SKA1', Tins=28, tobsyears=2, Sarea=25000, Dmax=4000, Ddish=15, Ndish=190, numin=350e6, numax=1050e6, deltanu=20e6):   #check survey area for interferometric mode
    #SKAmid1Pourtsidou=HIExperiment(40, t2y, 4000, 0.3*1000**2, 473e6, 20e6)
    #print 'SKA MID 1 Noise (Poutsidou et al params): ', SKAmid1Pourtsidou.getThermalNoise()
    tobs=tobsyears*365.25*24*60*60
    SKAmid1Bull=HIExperiment(name, Tins, tobs, Sarea, Dmax, Ddish, Ndish, numin, numax, deltanu)
    print 'SKAmid1 object initialised with area calculated from specs listed in Bull et al 2014'
    return SKAmid1Bull

def getMeerkatobject(name='Meerkat', Tins=30, tobsyears=2, Sarea=25000, Dmax=4000, Ddish=13.5, Ndish=64, numin=580e6, numax=1015e6, deltanu=20e6):   #check survey area for interferometric mode
    tobs=tobsyears*365.25*24*60*60
    Meerkat=HIExperiment(name, Tins, tobs, Sarea, Dmax, Ddish, Ndish, numin, numax, deltanu)
    print 'Meerkat object initialised with area calculated from specs listed in Bull et al 2014'
    return Meerkat
"""

def getTsky(nu):
    return 66*(nu/(300e6))**(-2.55)

if __name__=='__main__':
    Meerkat_i=getHIExptObject('Meerkat')
    #SKAmid1_d=getHIExptObject('SKA1', 'dish')
    #Meerkat_d=getHIExptObject('Meerkat', 'dish')
    #CHIME=getHIExptObject('CHIME')
    #z_list = np.loadtxt('Warren_cosmological_results/z_list.dat')
    z=1.
    #z=z_list[0]
    nu=1420e6/(z+1.)
    nu_MHz=nu/1e6
    zbinning_struct = binning.setup_ZBinningStruct(HIRAX)
    zbin_index=2
    zbin_prop = binning.setup_ZBinProp(zbin_index, zbinning_struct, HIRAX)


    #print('Chi at z_i', z , 'is', cd.comoving_distance(z,**cosmo))

    for expt in [HIRAX]:#[Meerkat_i, SKAmid1_i, Meerkat_d, SKAmid1_d , HIRAX]:
        delta_D=np.sqrt(expt.getDishArea())
        delta_ell=delta_D*2*np.pi*nu/c
        i_max=np.ceil(expt.Dmax/delta_D)+1
        #ell=np.linspace(10., 4000, 500)
        #ell=np.linspace(delta_ell, delta_ell*i_max, i_max)
        ell=np.linspace(delta_ell, delta_ell*i_max, 1000)

        if expt.mode[0].lower()=='d':
            k_perp=np.logspace(np.log10(1e-3), 1, 101)
            ell=k_perp*r


        if expt.name=='Meerkat' and expt.mode[0].lower()=='i':
            ell=np.loadtxt('BaselineDataFiles/radiofisher_array_config/nx_MKREF2_dec90.dat')[:,0]*2*np.pi*nu_MHz
        elif expt.name=='SKA1' and expt.mode[0].lower()=='i':
            ell=np.loadtxt('BaselineDataFiles/radiofisher_array_config/nx_SKAMREF2_dec90.dat')[:,0]*2*np.pi*nu_MHz
        elif expt.name=='CHIME':
            ell=np.loadtxt('BaselineDataFiles/radiofisher_array_config/nx_CHIME_400.dat')[:,0]*2*np.pi*nu_MHz


        if expt.name=='HIRAX':
            use_Bull_n_x=False
        else:
            use_Bull_n_x=True
        noise=expt.getNoiseAngularPower(ell, nu,zbin_prop)
        noise2=expt.getPlottableNoise(ell,nu)
        n_l=expt.getNumDensityInUV(ell, nu)

        #n_uni=expt.getNumDensityUniform(ell, nu)

        print 'Noise', expt.name, ':', noise
        """
        plt.plot(ell, n_l, label=expt.name)
        #plt.plot(ell, n_uni)
        plt.xlabel('l')
        plt.ylabel('n(u)')
        plt.title('Baseline number density n(u), z='+str(z))
        plt.legend()
        plt.show()

        plt.loglog(ell, noise, abel=expt.name)
        plt.xlabel('l')
        plt.ylabel(r'$C_l^N [K^2]$')
        plt.title('noise angular power spectrum')
        plt.ylim(1e-12,1e-4)
        plt.legend()
        #plt.show()
        """
        r=cd.comoving_distance(z, **cosmo)
        k_perp=ell/r
        if use_Bull_n_x:
            plt.loglog(k_perp, noise2, label=expt.name+' '+expt.mode) #noise*expt.tobs*expt.deltanu
        else:
            plt.loglog(k_perp, noise2, label=expt.name+' '+expt.mode)
        plt.xlabel(r'$k_\perp$')
        plt.ylabel(r'$T^2_{sys}FOV/n(u)$')
        plt.title('noise power spectrum')
        plt.ylim(1e-1, 1e6)
        plt.legend(loc='lower right')

    plt.show()

    plt.plot(ell, n_l)
    plt.xlim([0,4000])
    #plt.plot(ell, n_uni)
    plt.xlabel('l')
    plt.ylabel('n(u=l/2pi)')
    plt.title('Baseline number density, z='+str(z))
    plt.show()
