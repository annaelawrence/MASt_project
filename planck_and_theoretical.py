import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats
import healpy as hp
from astropy.io import fits
import camb

############################## LIMBER APPROX ############################
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


# Get lensing window function (flat universe)
win = ((chistar - chis) / (chis ** 2 * chistar)) ** 2

b_c = 1
z_c = 2
sigma_z = 2
h = 6.62607004 * 10 ** -34
k_B = 1.38064852 * 10 ** -23
T = 34
nu = 545  # GHz ??
beta = 2
H0 = 67.5 / (3 * 10 ** 5)
omega_m = 0.3
c = 3 * 10 ** 8
r_H = 1.0 / H0

# Do integral over chi
ls = np.arange(2, 2500 + 1, dtype=np.float64)
cl_kappa = np.zeros(ls.shape)
w = np.ones(chis.shape)  # this is just used to set to zero k values out of range of interpolation
for i, l in enumerate(ls):
    k = (l + 0.5) / chis
    w[:] = 1
    w[k < 1e-4] = 0
    w[k >= kmax] = 0

    # b_c = 2(b - 1) * f_nl * delta_c * (3 * omega_m) / (2 * a * g_a * r_H**2 * k**2)

    window_CIB = (-2.0 / 3.0 * 1 / omega_m * 1 / H0 ** 2 * 1 / (1 + zs)) * b_c * (chis ** 2 / (1 + zs) ** 2) * np.exp(
        -(zs - z_c) ** 2 / (2 * sigma_z ** 2)) * (np.exp((h * nu) / k_B * T) - 1) ** -1 * nu ** (beta + 3)

    lensing_kernel_flat = -2 * (chistar - chis) / (chis * chistar)

    win = window_CIB * lensing_kernel_flat * 1 / chis ** 2

    cl_kappa[i] = np.dot(dchis, w * PK.P(zs, k, grid=False) * win / k ** 2)
cl_kappa *= ls * (ls + 1) / 2.0

# plotting cl vs l
cl_limber = ls * cl_kappa  # compare to plot in Planck paper (Fig. D1)
plt.figure()
plt.plot(ls, cl_limber, color='b')
plt.xlim([1, 2000])
plt.legend(['Limber', 'CAMB hybrid'])
plt.ylabel(r'$L C_L^{{\mathrm{CIB}} \times \kappa}$')
plt.xlabel('$L$')
plt.show()

##################### PLANCK PLOTS ##########################################
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp


N_side = 2048  # Healpix map has 12 * N_side**2 pixels


# Cosmic Infrared Background
CIB_data = hp.read_map('COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits')

hp.mollview(CIB_data, title='CIB map', unit='mK', norm='hist', min=-1, max=1, xsize=2000)
hp.graticule()

# CIB mask
CIB_mask = hp.read_map('COM_Mask_Lensing_2048_R2.00.fits')
fsky_CIB = np.sum(CIB_mask) * 1. / len(CIB_mask)


CIB_data = CIB_data - np.mean(CIB_data)
LMAX = 1024
cl_CIB = hp.anafast(CIB_data, lmax=LMAX)



# Full Planck 545GHz Survey
full_545_data = hp.read_map('HFI_SkyMap_545-field-Int_2048_R3.00_full.fits')

full_545_data = full_545_data - np.mean(full_545_data)
LMAX = 1024
cl_full_545 = hp.anafast(full_545_data, lmax=LMAX)
l = np.arange(len(cl_full_545))


# Lensed Cosmic Microwave Background
CMB_lensing_data_alm = hp.read_alm('dat_klm.fits')  # Load alm file of kappa map
CMB_lensing_data = hp.alm2map(CMB_lensing_data_alm, N_side)


# lensing mask
CMB_mask = hp.read_map('mask.fits')
CMB_lensing_masked = hp.ma(CMB_lensing_data)
CMB_lensing_masked.mask = np.logical_not(CMB_mask)

fsky_CMB = np.sum(CMB_mask) * 1. / len(CMB_mask)


LMAX = 1024
cl_CMB_lensing = hp.anafast(CMB_lensing_masked.filled(), lmax=LMAX)
ell = np.arange(len(cl_CMB_lensing))


# CIB lensed CMB Cross Correlation
correlated_cls = hp.anafast(CMB_lensing_masked.filled(), CIB_data, lmax=LMAX)
ell = np.arange(len(correlated_cls))


# full survey correlation with lensed CMB
correlated_cls_fullsurvey = hp.anafast(CMB_lensing_masked.filled(), full_545_data, lmax=LMAX)
#ell = np.arange(len(correlated_cls_fullsurvey))

print('correlations done')

# Binning for full survey comparison
# Number of bins and range
Nbins = 30
lmin = 10
lmax = 1024

bins = np.round(np.linspace(lmin, lmax, Nbins + 1))  # Bin edges
bins = bins.astype(int)

lcenterbin = np.zeros(len(bins) - 1)

binned_CMB = np.zeros(len(bins) - 1)

binned_full_survey = np.zeros(len(bins) - 1)
binned_corr_cls_fullsurvey = np.zeros(len(bins) - 1)
print('binning...')

for k in range(0, len(bins) - 1):
    lmaxvec = np.arange(bins[k], bins[k + 1], 1)
    for l in lmaxvec:
        binned_full_survey[k] += cl_full_545[l]
    binned_full_survey[k] = binned_full_survey[k] / len(lmaxvec)

for k in range(0, len(bins) - 1):
    lmaxvec = np.arange(bins[k], bins[k + 1], 1)
    lcenterbin[k] = np.round(0.5 * (bins[k] + bins[k + 1]))  # bin center
    for l in lmaxvec:
        binned_corr_cls_fullsurvey[k] += correlated_cls_fullsurvey[l]

    binned_corr_cls_fullsurvey[k] = binned_corr_cls_fullsurvey[k] / len(lmaxvec)

fsky_CMB = np.sum(CMB_mask) * 1. / len(CMB_mask)
CIB_mask = hp.read_map('COM_Mask_Lensing_2048_R2.00.fits')
fsky_CIB = np.sum(CIB_mask) * 1. / len(CIB_mask)
mask = CIB_mask * CMB_mask
fsky = np.sum(mask) * 1. / len(mask)


plt.figure()
#plt.plot(ell, correlated_cls)
plt.plot(np.linspace(0, 1024, Nbins), binned_corr_cls_fullsurvey)
plt.plot(ls, cl_limber*(10**-33))
#
plt.xlim(10, 1000)
# plt.ylim(-1*(10**-7), 1*(10**-7))
plt.title('binned correlated power spectrum')
plt.xlabel('$\ell$');
plt.ylabel('$C_\ell^{\kappa I}$')
plt.ylim(-0.000001, 0.000002)
plt.show()


# plt.figure()
# plt.plot(ell, correlated_cls_fullsurvey / (2 * np.pi * fsky), alpha=0.3)
# # plt.errorbar(np.linspace(0, 1024, Nbins), binned_corr_cls_fullsurvey/(2*np.pi*fsky), delta_cls, linestyle='None', marker='o', markersize=4.5, capsize=3)
# plt.plot(np.linspace(0, 1024, Nbins), binned_corr_cls_fullsurvey / (2 * np.pi * fsky))
# plt.xlabel('$\ell$')
# plt.ylabel('$C_\ell^{\kappa I}/2 \pi [MJy/Sr] $')
# plt.ylim(-0.000001, 0.000002)
# plt.show()






