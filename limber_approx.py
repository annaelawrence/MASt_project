import camb
import numpy as np
import matplotlib.pyplot as plt

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


# Get the matter power spectrum interpolation object (based on RectBivariateSpline).
# Here for lensing we want the power spectrum of the Weyl potential.
PK = camb.get_matter_power_interpolator(pars, nonlinear=True,
                                        hubble_units=False, k_hunit=False, kmax=kmax,
                                        var1=camb.model.Transfer_Weyl, var2=camb.model.Transfer_Weyl, zmax=zs[-1])


# Have a look at interpolated power spectrum results for a range of redshifts
# Expect linear potentials to decay a bit when Lambda becomes important, and change from non-linear growth
# plt.figure(1, figsize=(8, 5))
# k = np.exp(np.log(10) * np.linspace(-4, 2, 200))
# zplot = [0, 0.5, 1, 4, 20, 1100]
# for z in zplot:
#     plt.loglog(k, PK.P(z, k))
# plt.xlim([1e-4, kmax])
# plt.xlabel('k Mpc')
# plt.ylabel('$P_\Psi\, Mpc^{-3}$')
# plt.legend(['z=%s' % z for z in zplot])
# plt.show()

###################################

b_c = 1
z_c = 2
sigma_z = 2
h = 6.62607004 * 10 ** -34
k_B = 1.38064852 * 10 ** -23
T = 34
nu = 545 # GHz
beta = 2
H0=67.5/(3*10**5)
omega_m = 0.3
c = 3*10**8
r_H = 1.0/H0 # (HO already in Mpc-1)


# Do integral over chi
ls = np.arange(2, 2500 + 1, dtype=np.float64)
cl_kappa = np.zeros(ls.shape)
w = np.ones(chis.shape)  # this is just used to set to zero k values out of range of interpolation
for i, l in enumerate(ls):
    k = (l + 0.5) / chis
    w[:] = 1
    w[k < 1e-4] = 0
    w[k >= kmax] = 0
    
    #b_c = 2(b - 1) * f_nl * delta_c * (3 * omega_m) / (2 * a * g_a * r_H**2 * k**2)

    # CIB windown function for Weyl
    window_CIB = (-2.0/3.0 * 1/omega_m *1/H0**2 * 1/(1+zs)) * b_c * (chis ** 2 / (1 + zs) ** 2) * np.exp(-(zs - z_c) ** 2 / (2 * sigma_z ** 2)) * (np.exp ((h * nu) / k_B * T) -1) **-1 * nu ** (beta + 3)

    # Lensing window function for Weyl
    lensing_kernel_flat = -2 * (chistar - chis) / (chis * chistar)

    # total window
    win = window_CIB * lensing_kernel_flat * 1 / chis ** 2 
    
    cl_kappa[i] = np.dot(dchis, w * PK.P(zs, k, grid=False) * win / k ** 2)
cl_kappa *= ls*(ls+1)/2.0


# plotting kappa cls vs l
cl_limber = ls*cl_kappa   # compare to plot in Planck paper (Fig. D1)
plt.figure(2)
plt.plot(ls, cl_limber, color='b')
plt.xlim([1, 2000])
plt.legend(['Limber', 'CAMB hybrid'])
plt.ylabel(r'$L  C_L^{\mathrm{CIB \times \kappa}}$')
plt.xlabel('$L$')
plt.show()

