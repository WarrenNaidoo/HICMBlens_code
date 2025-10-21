#saves HIRAX baseline density datafile, add SKA, Meerkat?
import numpy as np
import matplotlib.pyplot as plt
import HI_experiments as HIExperiments

#datafile Bull et al for comparison
xn=np.loadtxt('BaselineDataFiles/radiofisher_array_config/nx_MKREF2_dec90.dat')
x_Bull=xn[:,0]
n_x_Bull=xn[:,1]
c=3e8
c_rescale=c/1e6 # This represents c=nu*lambda with nu in MHz units hence the 1e6 factor

z_A=1.0
nu=1420e6/(1+z_A)
c=3e8
wavelength=c/nu


#HIexpt=HIExperiments.getHIExptObject('Meerkat') #'Meerkat' or 'HIRAX'
HIexpt=HIExperiments.getHIExptObject('HIRAX') #'Meerkat' or 'HIRAX'

if HIexpt.name=='HIRAX':

    #declination=-31 #don't use this, should I? #in degrees, approximately (KAT7 is at -30.7, actual dec close to but not exactly latitude

    Nside = int(np.ceil(np.sqrt(HIexpt.Ndish)))
    print('Ndish',HIexpt.Ndish)
    Ddish = HIexpt.Ddish

    dish_positions_x=np.arange(0, Ddish*Nside, Ddish)
    dish_positions_y=np.arange(0, Ddish*Nside, Ddish)

    print('dish positions',dish_positions_x)

    x,y=np.meshgrid(dish_positions_x, dish_positions_y)

    num_dishes=Nside**2
    Dmin = Ddish
    delta_D = Ddish/wavelength # u = D/lambda => delta_u = delta_D/lamdba = Ddish/lambda (Bull et al.;)

elif HIexpt.name=='Meerkat':
     positions=np.loadtxt('BaselineDataFiles/MeerkatDishxymetresCSV.csv', delimiter=',')
     x=positions[:,0]
     y=positions[:,1]

     delta_x_Bull=x_Bull[1]-x_Bull[0]
     xmin_Bull=np.amin(x_Bull)
     xmax_Bull=np.amax(x_Bull)

     Dmin = xmin_Bull*c_rescale
     Dmax = xmax_Bull*c_rescale
     delta_D = delta_x_Bull*c_rescale
     num_dishes=int(HIexpt.Ndish)

plt.scatter(x,y)
plt.title('Dish positions')
if HIexpt.name=='HIRAX':
    plt.xlim(-6, )
    plt.ylim(-6, )
plt.show()



try:
    baselines=np.loadtxt('BaselineDataFiles/'+HIexpt.name+'_baselines_Lx_Ly_with_duplicates_and_zero_6m.txt')
#note: each baseline counted twice - one negative and one positive
#zero 'baseline' for dish with itself also counted
except:
    print 'creating baseline array, may take a while'
    if HIexpt.name=='HIRAX':
        positions=np.zeros((Nside,Nside,2))
        for i in range(Nside):
            for j in range(Nside):
                positions[i,j,0]=x[i,j]
                positions[i,j,1]=y[i,j]

        print positions.shape
        positions=np.reshape(positions, (Nside**2,2))

    baselines=np.zeros((num_dishes, num_dishes, 2))
    for i in range(num_dishes):
        for j in range(num_dishes):
            print positions[i,:]-positions[j,:]
            baselines[i,j,:]=positions[i,:]-positions[j,:]

    baselines=baselines.reshape(num_dishes**2,2)

    np.savetxt('BaselineDataFiles/'+HIexpt.name+'_baselines_Lx_Ly_with_duplicates_and_zero.txt',baselines)

print('baseline_pairs', baselines)
plt.scatter(baselines[:,0], baselines[:,1])
plt.xlabel('Lx')
plt.ylabel('Ly')
plt.title('instantaneous uv coverage as a function of physical distance')
plt.show()


baselines_mag=np.sqrt(baselines[:,0]**2+baselines[:,1]**2)  #in metres not wavelengths
baseline_max=np.amax(baselines_mag)
np.savetxt('BaselineDataFiles/'+HIexpt.name+'_baselines_magnitudes_metres_with_duplicates_and_zero.txt',baselines_mag)

bin_size=np.sqrt(HIexpt.getDishArea())    #delta_u=sqrt(A_dish)/lambda, this is delta_u*lambda
i_max=np.ceil(baseline_max/bin_size)+1

#bins=np.arange(bin_size, bin_size*i_max, bin_size) -->rather use linspace so you know you have right num data points
#bins=np.linspace(bin_size, bin_size*i_max, i_max)

bin_centers=np.arange(Dmin, np.ceil(baseline_max)+4, delta_D)
print('bin centers',bin_centers)

bins=np.append(bin_centers-delta_D/2, np.amax(bin_centers+delta_D/2))
N_D, bins=np.histogram(baselines_mag, bins) #L=physical separation

N_D/=2    #Each baseline is counted twice - dividing by two makes us count each once

print 'baseline density added up: ',np.sum(N_D)
print 'Theoretical baseline number (for ', num_dishes,' dishes): ', num_dishes*(num_dishes-1)/2

plt.hist(baselines_mag, bins)
plt.xlabel('D')
plt.ylabel('2n(D)')
plt.title('total baseline number as a function of physical baseline length')
plt.show()

n_D=N_D/(np.pi*(bins[1:]**2-bins[0:bins.shape[0]-1]**2))
n_D_2=N_D/(2*np.pi*bin_centers*(bins[1]-bins[0]))
plt.plot(bins[1:],n_D)
plt.title('Number Density of Baselines')
plt.xlabel('baseline length in metres')
plt.ylabel('n')
plt.show()

bn=np.zeros((bin_centers.shape[0], 2))
bn[:,0]=bin_centers
bn[:,1]=n_D

np.savetxt('BaselineDataFiles/n_D_'+HIexpt.name+'.dat', bn)





#assume uniform distribution to cross check magnitude
if HIexpt.name=='HIRAX':
    Dmax=np.amax(baselines_mag)
    Dmin=HIexpt.Ddish

    num_baselines=Nside**2*(Nside**2-1)/2 #Nside^2=1024 not 1000

    n=num_baselines/(np.pi*(Dmax**2-Dmin**2))

    n=np.array([n])
    np.savetxt('BaselineDataFiles/n_HIRAX_uniform.dat', n)


    #plt.plot(bins[1:],n_L)
    plt.plot(bins[1:],bins[1:]/bins[1:]*n)
    plt.xlabel('distance in m')
    plt.title('number density of baselines')
    plt.show()






u=x_Bull*nu/1e6
n_u=n_x_Bull/nu**2*(1e6)**2

u_H=bin_centers/wavelength
n_u_H=n_D*wavelength**2
n_u_H_2=n_D_2*wavelength**2
import scipy
Hnu = scipy.interpolate.interp1d(u_H,n_u_H_2,fill_value=0.0)

plt.plot(u_H, n_u_H, label=r'$dA=\pi((u+du)^2-u^2)$')
plt.plot(u_H, n_u_H_2, label=r'$dA=2 \pi u du$')
plt.xlabel('u')
plt.ylabel('n(u)')
plt.title('Heather baseline density Meerkat (z=1)')
plt.ylim(0,0.0006)
plt.legend()
plt.xscale('log')
plt.show()

plt.plot(u, n_u)
plt.xlabel('u')
plt.ylabel('n(u)')
plt.title('Bull baseline density Meerkat (z=1)')
plt.show()


plt.plot(u, n_u, label='Bull et al')
plt.plot(u_H, n_u_H_2, label='Heather')
print('u_H',u_H)
plt.xlabel('u')
plt.ylabel('n(u)')
plt.title('Baseline density Meerkat (z=1)')
plt.legend()
plt.ylim(0,0.0012)
plt.show()

plt.plot(u, n_u, label='Bull et al')
#plt.plot(u_H, n_u_H_2, label='Heather')
plt.plot(u_H,n_u_H)
plt.xscale('log')
plt.xlabel('u')
plt.ylabel('n(u)')
plt.title('Baseline density Meerkat (z=1)')
plt.legend()
#plt.show()
print('u Heather', u_H)
#devin_n_u = np.load('devin_n_u_hirax.npz')
#plt.plot(devin_n_u['arr_0'],devin_n_u['arr_1'])
#print('u',devin_n_u['arr_0'])
#plt.show()
delta_x=x[1]-x[0]
print delta_x*c/1e6
print np.sqrt(HIexpt.getDishArea())

du=u[1]-u[0]

integrand_Bull=2*np.pi*u*du*n_u
integrand_H=2*np.pi*u_H*du*n_u_H_2
#integrand_H_2=(np.pi*u_H[1:]**2-np.pi*u_H[0:u_H.shape[0]-1]**2)*n_u_H

num_Bull=np.sum(integrand_Bull)
num_H=np.sum(integrand_H)
#num_H_2=np.sum(integrand_H_2)

print 'number of baselines Bull:', num_Bull
print 'number of baselines Heather:', num_H

#print 'number of baselines Heather full area:', num_H_2

D_Bull=x_Bull*c_rescale
n_D_Bull=n_x_Bull/c_rescale**2.
Dn=np.zeros((D_Bull.shape[0],2))
Dn[:,0]=D_Bull
Dn[:,1]=n_D_Bull
np.savetxt('BaselineDataFiles/n_D_'+str(HIexpt.Ndish)+HIexpt.name+'_Bull.dat', Dn)
