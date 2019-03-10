
# coding: utf-8

# In[1]:


import camb
import numpy as np
import matplotlib.pyplot as plt
import scipy


# In[2]:


# For calculating large-scale structure and lensing results yourself, get a power spectrum
# interpolation object. In this example we calculate the CMB lensing potential power
# spectrum using the Limber approximation, using PK=camb.get_matter_power_interpolator() function.
# calling PK(z, k) will then get power spectrum at any k and redshift z in range.

nz = 100  # number of steps to use for the radial/redshift integration
kmax = 10  # kmax to use
# First set up parameters as usual
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)


# In[3]:


# For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
# so get background results to find chistar, set up arrage in chi, and calculate corresponding redshifts
results = camb.get_background(pars)
chistar = results.conformal_time(0) - results.tau_maxvis
chis = np.linspace(0, chistar, nz)
zs = results.redshift_at_comoving_radial_distance(chis)
# Calculate array of delta_chi, and drop first and last points where things go singular
dchis = (chis[2:] - chis[:-2]) / 2
chis = chis[1:-1]
zs = zs[1:-1]


# In[4]:


# Get the matter power spectrum interpolation object (based on RectBivariateSpline).
# Here for lensing we want the power spectrum of the Weyl potential.
PK = camb.get_matter_power_interpolator(pars, nonlinear=True,
                                        hubble_units=False, k_hunit=False, kmax=kmax,
                                        var1=camb.model.Transfer_Weyl, var2=camb.model.Transfer_Weyl, zmax=zs[-1])


# In[5]:


# Have a look at interpolated power spectrum results for a range of redshifts
# Expect linear potentials to decay a bit when Lambda becomes important, and change from non-linear growth
plt.figure(1, figsize=(8, 5))
k = np.exp(np.log(10) * np.linspace(-4, 2, 200))
zplot = [0, 0.5, 1, 4, 20, 1100]
for z in zplot:
    plt.loglog(k, PK.P(z, k))
plt.xlim([1e-4, kmax])
plt.xlabel('k Mpc')
plt.ylabel('$P_\Psi\, Mpc^{-3}$')
plt.legend(['z=%s' % z for z in zplot])
plt.savefig('theoretical plots/power spectrum for redshift 0-1100.png')
plt.show()


# In[6]:


len(chis)


# In[7]:


# Get lensing window function (flat universe)
win = ((chistar-chis)/(chis**2*chistar))**2

a=1/(1+zs)
b_c = 1
z_c = 2
sigma_z = 2
h = 6.62607004 * 10 ** -34
k_B = 1.38064852 * 10 ** -23
T = 34
nu = 545 #GHz ??
beta = 2
H0=67.5/(3*10**5)
omega_m = 0.3
c = 3*10**8
r_H = 1.0/H0

b=2
delta_c=178
f_nl=1

# Do integral over chi
ls = np.arange(2, 2500 + 1, dtype=np.float64)
cl_kappa = np.zeros(ls.shape)
cl_kappa_CIB = np.zeros(ls.shape)
w = np.ones(chis.shape)  # this is just used to set to zero k values out of range of interpolation
for i, l in enumerate(ls):
    k = (l + 0.5) / chis
    w[:] = 1
    w[k < 1e-4] = 0
    w[k >= kmax] = 0
    
    g_a = []
    for j in np.arange(0, len(zs)):
        grow_int = lambda z: 1/((1/(1+z))*results.h_of_z(z)/H0)**3 * (1/(1+z)**2)
    
        g_a.append(scipy.integrate.quad(grow_int, 0, zs[j])[0])
    print(g_a)
    
    b_c = 2*(b - 1) * f_nl * delta_c * (3 * omega_m) / (2 * a * g_a * r_H**2 * k**2)
    
    window_CIB = (-2.0/3.0 * 1/omega_m *1/H0**2 * 1/(1+zs)) * b_c * (chis ** 2 / (1 + zs) ** 2) * np.exp(-(zs - z_c) ** 2 / (2 * sigma_z ** 2)) * (np.exp ((h * nu) / k_B * T) -1) **-1 * nu ** (beta + 3)
    
    lensing_kernel_flat = -2 * (chistar - chis) / (chis * chistar)

    win = window_CIB * lensing_kernel_flat * 1 / chis ** 2 
    
    CIB_auto_window = window_CIB**2 / chis**2
    
    cl_kappa[i] = np.dot(dchis, w * PK.P(zs, k, grid=False) * win / k ** 2)
    
    cl_kappa_CIB[i] = np.dot(dchis, w * PK.P(zs, k, grid=False) * CIB_auto_window / k ** 2)

cl_kappa *= ls*(ls+1)/2.0

cl_kappa_CIB *= ls*(ls+1)/2.0



# In[9]:


# plotting cl vs l
cl_limber = cl_kappa   # compare to plot in Planck paper (Fig. D1)
plt.figure(2)
plt.plot(ls, cl_limber, color='b')
plt.xlim([1, 2000])
plt.legend(['Limber', 'CAMB hybrid'])
plt.ylabel(r'$C_L^{\mathrm{CIB \times \kappa}}$')
plt.xlabel('$L$')
#plt.savefig('theoretical plots/cl CIBxkappa.png')
plt.show()


# In[10]:


# plotting l cl vs l
cl_limber = cl_kappa   # compare to plot in Planck paper (Fig. D1)
plt.figure(3)
plt.plot(ls, ls*cl_limber, color='b')
plt.xlim([1, 2000])
plt.legend(['Limber', 'CAMB hybrid'])
plt.ylabel(r'$L  C_L^{\mathrm{CIB \times \kappa}}$')
plt.xlabel('$L$')
#plt.savefig('theoretical plots/l cl CIBxkappa.png')
plt.show()


# In[11]:


# Get lensing window function (flat universe)
win = ((chistar-chis)/(chis**2*chistar))**2

a=1/(1+zs)
b_c = 1
z_c = 2
sigma_z = 2
h = 6.62607004 * 10 ** -34
k_B = 1.38064852 * 10 ** -23
T = 34
nu = 545 #GHz ??
beta = 2
H0=67.5/(3*10**5)
omega_m = 0.3
c = 3*10**8
r_H = 1.0/H0

b=2
delta_c=1
f_nl=1

# Do integral over chi
ls = np.arange(2, 2500 + 1, dtype=np.float64)
cl_kappa = np.zeros(ls.shape)
cl_kappa_CIB = np.zeros(ls.shape)
w = np.ones(chis.shape)  # this is just used to set to zero k values out of range of interpolation
for i, l in enumerate(ls):
    k = (l + 0.5) / chis
    w[:] = 1
    w[k < 1e-4] = 0
    w[k >= kmax] = 0
    
#     g_a = []
#     for j in np.arange(0, len(zs)):
#         grow_int = lambda z: 1/((1/(1+z))*results.h_of_z(z)/H0)**3 * (1/(1+z)**2)
    
#         g_a.append(scipy.integrate.quad(grow_int, 0, zs[j])[0])
#     print(g_a)
    
#     b_c = 2*(b - 1) * f_nl * delta_c * (3 * omega_m) / (2 * a * g_a * r_H**2 * k**2)
    
    window_CIB = (-2.0/3.0 * 1/omega_m *1/H0**2 * 1/(1+zs)) * b_c * (chis ** 2 / (1 + zs) ** 2) * np.exp(-(zs - z_c) ** 2 / (2 * sigma_z ** 2)) * (np.exp ((h * nu) / k_B * T) -1) **-1 * nu ** (beta + 3)
    
    lensing_kernel_flat = -2 * (chistar - chis) / (chis * chistar)

    win = window_CIB * lensing_kernel_flat * 1 / chis ** 2 
    
    CIB_auto_window = window_CIB**2 / chis**2
    
    cl_kappa[i] = np.dot(dchis, w * PK.P(zs, k, grid=False) * win / k ** 2)
    
    cl_kappa_CIB[i] = np.dot(dchis, w * PK.P(zs, k, grid=False) * CIB_auto_window / k ** 2)

cl_kappa *= ls*(ls+1)/2.0

cl_kappa_CIB *= ls*(ls+1)/2.0



# In[12]:


# plotting cl vs l
cl_limber = cl_kappa   # compare to plot in Planck paper (Fig. D1)
plt.figure(2)
plt.plot(ls, cl_limber, color='b')
plt.xlim([1, 2000])
plt.legend(['Limber', 'CAMB hybrid'])
plt.ylabel(r'$C_L^{\mathrm{CIB \times \kappa}}$')
plt.xlabel('$L$')
#plt.savefig('theoretical plots/cl CIBxkappa.png')
plt.show()


# In[13]:


# plotting l cl vs l
cl_limber = cl_kappa   # compare to plot in Planck paper (Fig. D1)
plt.figure(3)
plt.plot(ls, ls*cl_limber, color='b')
plt.xlim([1, 2000])
plt.legend(['Limber', 'CAMB hybrid'])
plt.ylabel(r'$L  C_L^{\mathrm{CIB \times \kappa}}$')
plt.xlabel('$L$')
#plt.savefig('theoretical plots/l cl CIBxkappa.png')
plt.show()


# In[43]:


from astropy.table import Table


# In[44]:


bin_t = Table.read('MASt_project-master/MASt_project-master/binned data.txt', format='ascii')
binned_correlation = bin_t['binned_corr_cls']
binned_CIB = bin_t['binned_CIB']


# In[45]:


delta_t = Table.read('MASt_project-master/MASt_project-master/error data.txt', format='ascii')
correlation_error = delta_t['delta_cls']
CIB_error = delta_t['delta_cls_CIB']


# In[46]:


data_t = Table.read('MASt_project-master/MASt_project-master/cl data.txt', format='ascii')
correlated_cl = data_t['correlated_cls']
CIB_data = data_t['cl_CIB']
ell = data_t['ell']


# In[47]:


more_binned_t = Table.read('MASt_project-master/MASt_project-master/CIB data more bins.txt', format='ascii')
# binned_correlation = bin_t['']
binned_CIB = more_binned_t['binned_CIB_morebins']
CIB_error = more_binned_t['delta_cls_CIB_morebins']


# In[41]:


fsky = 0.60005615154902137
fsky_CIB = 0.63341087102890015


# In[20]:


Nbins = 25


# In[69]:


plt.figure(3)
plt.plot(ls, cl_limber*8*10**-32, color='r')
plt.plot(ell, correlated_cl/ (2 * np.pi * fsky), alpha=0.3)
plt.errorbar(np.linspace(0, 1024, Nbins), binned_correlation/ (2 * np.pi * fsky), correlation_error,
            linestyle='None', marker='o', markersize=4.5, capsize=3)
plt.xlim([1, 1050])
plt.legend(['Limber', 'Planck results', 'errors'])
plt.ylabel(r'$C_L^{\mathrm{CIB \times \kappa}}$')
plt.xlabel('$L$')
plt.ylim(-0.000000003, 0.2*(10**-7))
#plt.savefig('theoretical plots/cl CIBxkappa with planck (fsky).png')
plt.show()


# In[55]:


plt.figure(4)
plt.plot(ls, ls*cl_limber*8*10**-32, color='r')
plt.plot(ell, ell*correlated_cl/ (2 * np.pi * fsky), alpha=0.3)
plt.errorbar(np.linspace(0, 1024, Nbins), np.linspace(0, 1024, Nbins)*binned_correlation/ (2 * np.pi * fsky), np.linspace(0, 1024, Nbins)*correlation_error,
            linestyle='None', marker='o', markersize=4.5, capsize=3)
plt.xlim([1, 1050])
plt.legend(['Limber', 'Planck results', 'errors'])
plt.ylabel(r'$L  C_L^{\mathrm{CIB \times \kappa}}$')
plt.xlabel('$L$')
plt.ylim(-0.01*(10**-4), 2.5*(10**-6))
plt.savefig('theoretical plots/l cl CIBxkappa with planck (fsky).png')
plt.show()


# ## CIB auto spectrum

# In[56]:


# plotting cl vs l
cl_limber_CIB = cl_kappa_CIB   # compare to plot in Planck paper (Fig. D1)
plt.figure(5)
plt.plot(ls, cl_limber_CIB, color='b')
plt.xlim([1, 2000])
plt.legend(['Limber', 'CAMB hybrid'])
plt.ylabel(r'$C_L^{\mathrm{CIB \times CIB}}$')
plt.xlabel('$L$')
plt.savefig('theoretical plots/cl CIBxCIB.png')
plt.show()


# In[57]:


plt.figure(6)
plt.plot(ls, ls*cl_limber_CIB, color='b')
plt.xlim([1, 2000])
plt.legend(['Limber', 'CAMB hybrid'])
plt.ylabel(r'$L  C_L^{\mathrm{CIB \times CIB}}$')
plt.xlabel('$L$')
plt.savefig('theoretical plots/l cl CIBxCIB.png')
plt.show()


# In[61]:


plt.figure(7)
plt.plot(ls, cl_limber_CIB*1.5*10**-68, color='r')
plt.plot(ell, CIB_data/ (2 * np.pi * fsky_CIB), alpha=0.3)
plt.errorbar(np.linspace(0, 1024, 80)[1::3], (binned_CIB/ (2 * np.pi * fsky_CIB))[1::3], CIB_error[1::3],
            linestyle='None', marker='o', markersize=4.5, capsize=3, color='orange')
plt.xlim([1, 1000])
plt.legend(['Limber', 'Planck results', 'errors'])
plt.ylabel(r'$C_L^{\mathrm{CIB \times CIB}}$')
plt.xlabel('$L$')
plt.ylim(-0.05*(10**-8), 0.2*(10**-7))
plt.savefig('theoretical plots/cl CIBxCIB with planck (fsky).png')
plt.show()


# In[66]:


plt.figure(8)
plt.plot(ls, ls*cl_limber_CIB*1.6*10**-68, color='r')
plt.plot(ell, ell*CIB_data/ (2 * np.pi * fsky_CIB), alpha=0.3)
plt.errorbar(np.linspace(0, 1024, 80)[1::3], (np.linspace(0, 1024, 80)*binned_CIB/ (2 * np.pi * fsky_CIB))[1::3], (np.linspace(0, 1024, 80)*CIB_error)[1::3],
            linestyle='None', marker='o', markersize=4.5, capsize=3, color='orange')
plt.xlim([1, 1050])
plt.legend(['Limber', 'Planck results', 'errors'])
plt.ylabel(r'$L  C_L^{\mathrm{CIB \times CIB}}$')
plt.xlabel('$L$')
plt.ylim(-0.01*(10**-5), 0.35*(10**-5))
plt.savefig('theoretical plots/l cl CIBxCIB with planck (fsky).png')
plt.show()


# In[68]:


np.sqrt(1.6*10**-68)

