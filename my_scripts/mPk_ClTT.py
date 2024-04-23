#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary modules
# uncomment to get plots displayed in notebook
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.optimize import fsolve
import math


# In[ ]:


########################################################
#
# Constraints to be matched
#
# As explained in the "Neutrino cosmology" book, CUP, Lesgourgues et al., section 5.3, the goal is to vary
# - omega_cdm by a factor alpha = (1 + coeff*Neff)/(1 + coeff*3.046)
# - h by a factor sqrt*(alpha)
# in order to keep a fixed z_equality(R/M) and z_equality(M/Lambda)
#
omega_b = 0.0223828
omega_cdm_standard = 0.1201075
h_standard = 0.67810
#
# coefficient such that omega_r = omega_gamma (1 + coeff*Neff),
# i.e. such that omega_ur = omega_gamma * coeff * Neff:
# coeff = omega_ur/omega_gamma/Neff_standard
# We could extract omega_ur and omega_gamma on-the-fly within the script,
# but for simplicity we did a preliminary interactive run with background_verbose=2
# and we copied the values given in the budget output.
#
coeff = 1.70961e-05/2.47298e-05/3.039
print ("coeff=",coeff)          
#
#############################################
#
# Fixed settings
#
common_settings = {# fixed LambdaCDM parameters
                   'omega_b':omega_b,
                   'A_s':2.100549e-09,
                   'n_s':0.9660499,
                   'tau_reio':0.05430842,
                   # output and precision parameters
                   'output':'tCl,pCl,lCl,mPk',
                   'lensing':'yes',
                   'P_k_max_1/Mpc':3.0,
                   'l_switch_limber':9}
#
##############################################
#

# loop over varying parameter values
#
M = {}
#
#setting Delta N_eff ~= 0.01
N_ur = 3.04895  #diff. temp. thermal
#N_ur = 3.04973   # DW                    
#
# rescale omega_cdm and h
#
alpha = (1.+coeff*N_ur)/(1.+coeff*3.039)
omega_cdm = (omega_b + omega_cdm_standard)*alpha - omega_b              
h = h_standard*math.sqrt(alpha)                                     
print (' * Compute with %s=%e, %s=%e, %s=%e'%('N_ur',N_ur,'omega_cdm',omega_cdm,'h',h))
#
# call CLASS
#
M[0] = Class()
M[0].set({'omega_b':0.0223828,'omega_cdm':0.1201075,'h':0.67810,'A_s':2.100549e-09,'n_s':0.9660499,'tau_reio':0.05430842})
M[0].set({'output':'tCl,pCl,lCl,mPk','lensing':'yes','P_k_max_1/Mpc':3.0})
# run class
M[0].compute()
#
M[1] = Class()
M[1].set(common_settings)
M[1].set({'N_ur':N_ur})
M[1].set({'omega_cdm':omega_cdm})
M[1].set({'h':h})
# run class
M[1].compute()


# In[ ]:


# esthetic definitions for the plots
font = {'size'   : 16, 'family':'STIXGeneral'}
axislabelfontsize='small'
matplotlib.rc('font', **font)
#matplotlib.mathtext.rcParams['legend.fontsize']='medium'
#plt.rcParams["figure.figsize"] = [8.0,6.0]


# In[ ]:


#############################################
#
# extract spectra and plot them
#
#############################################
kvec = np.logspace(-4,np.log10(3),1000) # array of kvec in h/Mpc
twopi = 2.*math.pi
#
# Create figures
#
fig_Pk, ax_Pk = plt.subplots()
fig_TT, ax_TT = plt.subplots()
#
# loop over varying parameter values
#
ll = {}
clM = {}
clTT = {}
pkM = {}
clphiphi = {}
legarray = []


for i in range(2):

#
# get Cls
#
    clM[i] = M[i].lensed_cl(2500)
    ll[i] = clM[i]['ell'][2:]
    clTT[i] = clM[i]['tt'][2:]
#
# store P(k) for common k values
#
    pkM[i] = []
# The function .pk(k,z) wants k in 1/Mpc so we must convert kvec for each case with the right h
    if i==0:
        h=M[0].h()
    else:
        alpha = (1.+coeff*N_ur)/(1.+coeff*3.039)
        h = 0.67810*math.sqrt(alpha) # this is h
    
    khvec = kvec*h # This is k in 1/Mpc
    for kh in khvec:
        pkM[i].append(M[i].pk(kh,0.)*h**3)


#Plotting the ratio of matter power spectrum for our distribution and the standard thermal distribution
#ax_Pk.semilogx(kvec,np.array(pkM[1])/np.array(pkM[0]),'r-',label=r'ratio $P(k)$')
#Plotting the relative difference
#ax_Pk.set_ylim([-0.0001,0.0008])
ax_Pk.semilogx(kvec,np.array(pkM[1])/np.array(pkM[0]),'b-',label=r'$m_s=3.1627~\mathrm{eV}, \Delta N_\mathrm{eff}=0.00935$')
ax_Pk.legend(loc='upper left')
ax_Pk.set_xlabel(r'$k \,\,\,\, [h/\mathrm{Mpc}]$')
#ax_Pk.set_ylabel(r'$P(k)_{DW(m_s=10~\mathrm{eV})}/P(k)_{\Lambda CDM}$')
ax_Pk.set_ylabel(r'$P(k)/P(k)_{\Lambda CDM}$')
ax_Pk.set_title(r'diff. temp. thermal distribution')
fig_Pk.tight_layout()
fig_Pk.savefig('Pk_difftempth.jpg')
#Plotting the ratio of ClTT for our distribution and the standard thermal distribution
#ax_TT.semilogx(ll[0],clTT[1]/clTT[0],'g-',label=r'ratio $C_\ell^{TT}$')
#Plotting the relative difference
ax_TT.semilogx(ll[0],clTT[1]/clTT[0],'r-',label=r'$m_s=3.1627~\mathrm{eV}, \Delta N_\mathrm{eff}=0.00935$')
ax_TT.legend(loc='lower left')
ax_TT.set_xlabel(r'$\mathrm{Multipole} \,\,\,\,  \ell$')
#ax_Pk.set_ylabel(r'$(C_\ell^{TT})_{DW(m_s=10~\mathrm{eV})}/(C_\ell^{TT})_{\Lambda CDM}$')
ax_TT.set_ylabel(r'$(C_\ell^{TT})/(C_\ell^{TT})_{\Lambda CDM}$')
ax_TT.set_title(r'diff. temp. thermal distribution')
fig_TT.tight_layout()
fig_TT.savefig('ClTT_difftempth.jpg')

with open('difftempth_Pk.txt', 'w') as output_file:
    for x, y in zip(kvec, np.array(pkM[1])):
        output_file.write(f'{x} {y}\n')