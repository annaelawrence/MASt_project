# Initialization
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats
import healpy as hp
from astropy.io import fits

N_side = 2048  # Healpix map has 12 * N_side**2 pixels

# Cosmic Infrared Background
CIB_data = hp.read_map('COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits')

hp.mollview(CIB_data, title='CIB map', unit='mK', norm='hist', min=-1, max=1, xsize=2000)
hp.graticule()

# CIB mask
CIB_mask = hp.read_map('COM_Mask_Lensing_2048_R2.00.fits')
fsky_CIB = np.sum(CIB_mask) * 1. / len(CIB_mask)
# hp.mollview(CIB_mask, title='CIB mask', norm='hist', xsize=2000)


CIB_data = CIB_data - np.mean(CIB_data)
LMAX = 1024
cl_CIB = hp.anafast(CIB_data, lmax=LMAX)
l = np.arange(len(cl_CIB))

plt.figure()
plt.plot(l, cl_CIB / (2 * np.pi))
plt.xlabel('$\ell$');
plt.ylabel('$C_{\ell}^{II} \2\pi$');
plt.grid()
plt.title('CIB power spectrum')




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

# Cross Correlation

correlated_cls = hp.anafast(CMB_lensing_masked.filled(), CIB_data, lmax=LMAX)
ell = np.arange(len(correlated_cls))

plt.figure()
plt.plot(ell, correlated_cls / (2 * np.pi))
plt.ylim(-0.00000002, 0.00000002)
plt.xlabel('$\ell$');
plt.ylabel('$C_\ell^{\kappa I} /2\pi$');
plt.grid()
plt.title('CIB-lensed CMB correlation power spectrum')

# ### Mask
print(fsky_CMB, fsky_CIB)

# Combined mask
mask = CIB_mask * CMB_mask
fsky = np.sum(mask) * 1. / len(mask)
print(fsky)

# Binning
# Number of bins and range
Nbins = 25
lmin = 10
lmax = 1024

print('Binning...')
bins = np.round(np.linspace(lmin, lmax, Nbins + 1))  # Bin edges
bins = bins.astype(int)

lcenterbin = np.zeros(len(bins) - 1)

binned_CIB = np.zeros(len(bins) - 1)
binned_CMB = np.zeros(len(bins) - 1)
binned_corr_cls = np.zeros(len(bins) - 1)

for k in range(0, len(bins) - 1):
    lmaxvec = np.arange(bins[k], bins[k + 1], 1)

    for l in lmaxvec:
        binned_CIB[k] += cl_CIB[l]
    binned_CIB[k] = binned_CIB[k] / len(lmaxvec)

for k in range(0, len(bins) - 1):
    lmaxvec = np.arange(bins[k], bins[k + 1], 1)
    for l in lmaxvec:
        binned_CMB[k] += cl_CMB_lensing[l]
    binned_CMB[k] = binned_CMB[k] / len(lmaxvec)

for k in range(0, len(bins) - 1):
    lmaxvec = np.arange(bins[k], bins[k + 1], 1)
    lcenterbin[k] = np.round(0.5 * (bins[k] + bins[k + 1]))  # bin center
    for l in lmaxvec:
        binned_corr_cls[k] += correlated_cls[l]

    binned_corr_cls[k] = binned_corr_cls[k] / len(lmaxvec)

print(binned_corr_cls)

# Binned Plots

plt.figure()
# plt.plot(ell, correlated_cls)
plt.plot(np.linspace(0, 1024, Nbins), binned_corr_cls)

plt.xlim(10, 1000)
# plt.ylim(-1*(10**-7), 1*(10**-7))
plt.title('binned correlated power spectrum')
plt.xlabel('$\ell$');
plt.ylabel('$C_\ell^{\kappa I}$')

# Errors

# Correlated errors


delta_cls = np.zeros(len(bins) - 1)
len(delta_cls)

for k in range(0, len(bins) - 1):
    lmaxvec = np.arange(bins[k], bins[k + 1], 1)

    for l in lmaxvec:
        delta_cls[k] += fsky * (2. * l + 1.) / (
                (cl_CMB_lensing[l] / fsky_CMB) * (cl_CIB[l] / fsky_CIB) + correlated_cls[l] ** 2)
    delta_cls[k] = 1. / delta_cls[k]
delta_cls = np.sqrt(delta_cls)

# CMB errors

delta_cls_CMB = np.zeros(len(bins) - 1)
len(delta_cls_CMB)

for k in range(0, len(bins) - 1):
    lmaxvec = np.arange(bins[k], bins[k + 1], 1)

    for l in lmaxvec:
        delta_cls_CMB[k] += fsky * (2. * l + 1.) / (2 * (cl_CMB_lensing[l] / fsky_CMB) ** 2)
    delta_cls_CMB[k] = 1. / delta_cls_CMB[k]
delta_cls_CMB = np.sqrt(delta_cls_CMB)


# CIBerrors

delta_cls_CIB = np.zeros(len(bins) - 1)
len(delta_cls_CIB)

for k in range(0, len(bins) - 1):
    lmaxvec = np.arange(bins[k], bins[k + 1], 1)

    for l in lmaxvec:
        delta_cls_CIB[k] += fsky * (2. * l + 1.) / (2 * (cl_CIB[l] / fsky_CIB) ** 2)
    delta_cls_CIB[k] = 1. / delta_cls_CIB[k]
delta_cls_CIB = np.sqrt(delta_cls_CIB)

#
# # Final CIB-lensed CMB correlation plot
#
# plt.plot(ell, correlated_cls / (2 * np.pi * fsky), alpha=0.3)
# plt.errorbar(np.linspace(0, 1024, Nbins), binned_corr_cls / (2 * np.pi * fsky), delta_cls, linestyle='None', marker='o',
#              markersize=4.5, capsize=3)
# plt.plot(np.linspace(0, 1024, Nbins), binned_corr_cls / (2 * np.pi * fsky))
# plt.xlabel('$\ell$')
# plt.ylabel('$C_\ell^{\kappa I}/2 \pi [MJy/Sr] $')
# plt.ylim(-0.05 * (10 ** -7), 0.2 * (10 ** -7))
# # plt.savefig('Project/Correlated everything.png')
#
#
#
# plt.plot(ell, ell * correlated_cls / (2 * np.pi * fsky), alpha=0.3)
# plt.errorbar(np.linspace(0, 1024, Nbins), np.linspace(0, 1024, Nbins) * binned_corr_cls / (2 * np.pi * fsky), delta_cls,
#              linestyle='None', marker='o', markersize=4.5, capsize=3)
# plt.plot(np.linspace(0, 1024, Nbins), np.linspace(0, 1024, Nbins) * binned_corr_cls / (2 * np.pi * fsky))
# plt.xlabel('$\ell$')
# plt.ylabel('$C_\ell^{\kappa I}/2 \pi [MJy/Sr] $')
# # plt.ylim(-0.05*(10**-7), 0.2*(10**-7))
# # plt.savefig('Project/Correlated everything.png')
#
#
# # CIB power spectrum
#
# plt.plot(ell, cl_CIB / (2 * np.pi * fsky_CIB), label='CIB', alpha=0.3)
# plt.errorbar(np.linspace(0, 1024, Nbins), binned_CIB / (2 * np.pi * fsky_CIB), delta_cls_CIB, linestyle='None',
#              marker='o', markersize=4.5, capsize=3)
# plt.plot(np.linspace(0, 1024, Nbins), binned_CIB / (2 * np.pi * fsky_CIB), label='Binned CIB')
# plt.xlabel('$\ell$')
# plt.ylabel('$C_\ell^{II}/2\pi$ [MJy/Sr]')
# plt.legend(loc='best')
# plt.ylim(0, 0.2 * (10 ** -7))
# # plt.savefig('Project/CIB everything.png')
#
#
# # Lensed CMB power spectrum
#
# plt.plot(ell, cl_CMB_lensing / (2 * np.pi * fsky_CMB), label='Lensed CMB', alpha=0.3)
# plt.errorbar(np.linspace(0, 1024, Nbins), binned_CMB / (2 * np.pi * fsky_CMB), delta_cls_CMB, linestyle='None',
#              marker='o', markersize=4.5, capsize=4)
# plt.plot(np.linspace(0, 1024, Nbins), binned_CMB / (2 * np.pi * fsky_CMB), label='Binned Lensed CMB')
# # plt.ylim(0, 5*(10**-7))
#
# plt.xlabel('$\ell$')
# plt.ylabel('$C_\ell^{\kappa \kappa}/2\pi$')
# plt.legend(loc='best')
# plt.ylim(1.5 * (10 ** -7), 4.5 * (10 ** -7))
# # plt.savefig('Project/2pi CMB plot.png', bbox_inches = "tight")
# # plt.savefig('Project/CMB everything.png')


from astropy.io import ascii
from astropy.table import Table

binned_data_table = Table([binned_CIB, binned_CMB, binned_corr_cls], names=('binned_CIB', 'binned_CMB', 'binned_corr_cls'))
data_table = Table([cl_CMB_lensing, cl_CIB, correlated_cls, ell], names=('cl_CMB_lensing', 'cl_CIB', 'correlated_cls', 'ell'))
error_table = Table([delta_cls_CMB, delta_cls, delta_cls_CIB], names=('delta_cls_CMB', 'delta_cls', 'delta_cls_CIB'))

ascii.write(binned_data_table, 'binned data.txt')
ascii.write(data_table, 'cl data.txt')
ascii.write(error_table, 'error data.txt')
