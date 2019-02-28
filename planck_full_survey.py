import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats
import healpy as hp
from astropy.io import fits

N_side = 2048  # Healpix map has 12 * N_side**2 pixels

# Full Planck 545GHz Survey
full_545_data = hp.read_map('HFI_SkyMap_545-field-Int_2048_R3.00_full.fits')
# hp.mollview(full_545_data, title='full survey map', unit='mK', norm='hist', min=-1,max=1, xsize=2000)

full_545_data = full_545_data - np.mean(full_545_data)
LMAX = 1024
cl_full_545 = hp.anafast(full_545_data, lmax=LMAX)
l = np.arange(len(cl_full_545))

plt.figure()
plt.plot(l, ((l ** 2) * cl_full_545) / (2 * np.pi))
plt.xlabel('$\ell$');
plt.ylabel('$\ell^2 \ c_{\ell} \ /2 \pi$');
plt.grid()
plt.title('Power Spectrum of full 545GHz Survey')


# Lensed Cosmic Microwave Background
# data_2 = hp.read_map('Project/COM_Lensing_4096_R3.00/TT/dat_klm.fits')

CMB_lensing_data_alm = hp.read_alm('dat_klm.fits')  # Load alm file of kappa map
CMB_lensing_data = hp.alm2map(CMB_lensing_data_alm, N_side)

# lensing map
hp.mollview(CMB_lensing_data, title='TT lensed CMB map', norm='hist', xsize=2000)
hp.graticule()

# lensing mask
CMB_mask = hp.read_map('mask.fits')
CMB_lensing_masked = hp.ma(CMB_lensing_data)
CMB_lensing_masked.mask = np.logical_not(CMB_mask)

fsky_CMB = np.sum(CMB_mask) * 1. / len(CMB_mask)
# hp.mollview(CMB_mask, title='Lensing mask', norm='hist', xsize=2000)


# filled lensing mask
hp.mollview(CMB_lensing_masked.filled(), title='Lensed CMB map', norm='hist', xsize=2000)


LMAX = 1024
cl_CMB_lensing = hp.anafast(CMB_lensing_masked.filled(), lmax=LMAX)
ell = np.arange(len(cl_CMB_lensing))

plt.figure()
plt.plot(ell, cl_CMB_lensing / (2 * np.pi))
plt.xlabel('$\ell$');
plt.ylabel('$C_{\ell}^{\kappa \kappa} /2\pi$');
plt.grid()
plt.title('Lensed CMB power spectrum')



# correlation with lensed CMB

correlated_cls_fullsurvey = hp.anafast(CMB_lensing_masked.filled(), full_545_data, lmax=LMAX)
ell = np.arange(len(correlated_cls_fullsurvey))

plt.figure()
plt.plot(ell, correlated_cls_fullsurvey)
plt.xlabel('$\ell$');
plt.ylabel('Cls');
plt.grid()
plt.ylim(-0.00025, 0.00025)

# Binning for full survey comparison
# Number of bins and range
Nbins = 25
lmin = 10
lmax = 1024

bins = np.round(np.linspace(lmin, lmax, Nbins + 1))  # Bin edges
bins = bins.astype(int)

lcenterbin = np.zeros(len(bins) - 1)

binned_CMB = np.zeros(len(bins) - 1)

binned_full_survey = np.zeros(len(bins) - 1)
binned_corr_cls_fullsurvey = np.zeros(len(bins) - 1)

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
# plt.plot(ell, correlated_cls)
plt.plot(np.linspace(0, 1024, Nbins), binned_corr_cls_fullsurvey)

plt.xlim(10, 1000)
# plt.ylim(-1*(10**-7), 1*(10**-7))
plt.title('binned correlated power spectrum')
plt.xlabel('$\ell$');
plt.ylabel('$C_\ell^{\kappa I}$')
plt.ylim(-0.000001, 0.000002)




plt.plot(ell, correlated_cls_fullsurvey / (2 * np.pi * fsky), alpha=0.3)
# plt.errorbar(np.linspace(0, 1024, Nbins), binned_corr_cls_fullsurvey/(2*np.pi*fsky), delta_cls, linestyle='None', marker='o', markersize=4.5, capsize=3)
plt.plot(np.linspace(0, 1024, Nbins), binned_corr_cls_fullsurvey / (2 * np.pi * fsky))
plt.xlabel('$\ell$')
plt.ylabel('$C_\ell^{\kappa I}/2 \pi [MJy/Sr] $')
plt.ylim(-0.00002, 0.00002)

plt.show()
