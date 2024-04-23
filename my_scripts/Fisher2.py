import numpy as np
import matplotlib.pyplot as plt
import math
from classy import Class 

def noise_exp():
    l_min = 2
    l_max = 3000
    f_sky = 0.4
    T_cmb_muK = 2.735*pow(10,-6)
    #Noise parameters
    sT = 33 * (np.pi/60./180.)*T_cmb_muK**2
    sP = sT * np.sqrt(2.)*T_cmb_muK**2
    theta_FWHM = 1.5 * (np.pi/60./180.)*T_cmb_muK

    NlTT = []
    NlEE = []
    NlPP = []
    for l in range(l_max+1):
        NlTT.append(sT**2*math.exp(l*(l+1)*theta_FWHM**2/(8*math.log(2))))
        NlEE.append(sP**2*math.exp(l*(l+1)*theta_FWHM**2/(8*math.log(2))))
        NlPP.append(0)                          #have to add noise for phi-phi
    Nl_dict = {'tt': NlTT, 'ee': NlEE, 'pp': NlPP}
    return [l_min,l_max,f_sky,Nl_dict]

def utility_function_call_CLASS(input_dict):
    #Compute Cl with this utility function, repeat less code.
    cosmo = Class()                                             #PROBLEM: Calls the default CLASS, not our modified CLASS
    cosmo.set(input_dict)
    cosmo.compute()
    l_max = noise_exp()[1]
    temp_cl = cosmo.lensed_cl(l_max)
    cosmo.struct_cleanup()
    cosmo.empty()
    return temp_cl

#Function to compute the derivatives
def get_deriv(params, theta, ch, theta_step):
    left_params = params.copy()
    left_params[theta] = params[theta] - theta_step
    right_params = params.copy()
    right_params[theta] = params[theta] + theta_step
    cl_left = utility_function_call_CLASS(left_params)[ch]
    cl_right = utility_function_call_CLASS(right_params)[ch]
    dCl_dtheta = (cl_right - cl_left) / (2 * theta_step)
    return dCl_dtheta

def return_deriv():

    #Arrays containing the derivatives of C_l's w.r.t all 7 parameters
    key_array = ['omega_b','omega_cdm','h','A_s','n_s','tau_reio','N_eff']
    step_array = [0.01,0.01,0.01,1e-10,0.1,0.01,0.01]
    npar = len(key_array)
    D_ClTT = [[] for _ in range(npar)]
    D_ClTE = [[] for _ in range(npar)]
    D_ClEE = [[] for _ in range(npar)]
    D_ClPP = [[] for _ in range(npar)]
    D_ClTP = [[] for _ in range(npar)]
    for i in range(npar):
        D_ClTT[i] = get_deriv( params,key_array[i],'tt',step_array[i])
        D_ClTE[i] = get_deriv( params,key_array[i],'te',step_array[i])
        D_ClEE[i] = get_deriv( params,key_array[i],'ee',step_array[i])
        D_ClPP[i] = get_deriv( params,key_array[i],'pp',step_array[i])
        D_ClTP[i] = get_deriv( params,key_array[i],'tp',step_array[i])
    DCl_dict = {'tt': D_ClTT, 'te': D_ClTE, 'ee': D_ClEE, 'pp': D_ClPP, 'tp': D_ClTP}
    return [npar,DCl_dict]



def fisher():
    # Constraints to be matched
    #
    # As explained in the "Neutrino cosmology" book, CUP, Lesgourgues et al., section 5.3, the goal is to vary
    # - omega_cdm by a factor alpha = (1 + coeff*Neff)/(1 + coeff*3.046)
    # - h by a factor sqrt*(alpha)
    # in order to keep a fixed z_equality(R/M) and z_equality(M/Lambda)
    #
    omega_b = 0.0223828
    omega_cdm_standard = 0.120108
    h_standard = 0.67810

    #coeff = omega_ur/omega_gamma/Neff_standard
    #Following values are obtained from a preliminary run of CLASS with background_verbose=2 and copied the values from ------Budget equation-------(terminal output)
    coeff = 3.71799e-5/5.37815e-5/3.039

    # rescale omega_cdm and h
    N_ur = 3.074    #Total N_eff from our non-thermal model.
    alpha = (1.+coeff*N_ur)/(1.+coeff*3.039)        #Contribution to N_eff from the 3 sterile neutrinos is 3.039 according to CLASS
    omega_cdm = (omega_b + omega_cdm_standard)*alpha - omega_b              
    h = h_standard*math.sqrt(alpha)


    # Define the CLASS input dictionary
    params = {
        'output': 'tCl lCl pCl',
        'ic': 'ad',
        'gauge': 'new',
        'l_max_scalars': 3000,
        'lensing': 'yes',
        'omega_b': omega_b, 
        'omega_cdm': omega_cdm,
        'h': h,
        'A_s':2.100549e-09,
        'n_s':0.9660499,
        'tau_reio':0.05430842,
        'N_eff':N_ur
        }
    print("omega_b =",omega_b,", omega_cdm =",omega_cdm,", h =",h,", N_eff =",N_ur)

    #Output from CLASS
    C_l = utility_function_call_CLASS(params)
    #Noise
    N_l = noise_exp()[3]
    
    npar = return_deriv()[0]
    DCl_list = [[] for _ in range(npar)]
    Cl_list = []
    invCl_list = []

    l_min = noise_exp()[0]
    l_max = noise_exp()[1]
    for l in range(l_max+1):
        Cl = np.array([[C_l['tt'][l]+N_l['tt'][l],C_l['te'][l],C_l['tp'][l]],[C_l['te'][l],C_l['ee'][l]+N_l['ee'][l],0],[C_l['tp'][l],0,C_l['pp'][l]+N_l['pp'][l]]])
        Cl_list.append(Cl)
        if (l>=l_min):
            invCl = np.linalg.inv(Cl)
        else:
            invCl = 0
        invCl_list.append(invCl)

    DCl = return_deriv()[1]
    for j in range(npar):
        for l in range(l_max+1):
            DCl_list[j].append(np.array([[DCl['tt'][j][l],DCl['te'][j][l],DCl['tp'][j][l]],[DCl['te'][j][l],DCl['ee'][j][l],0],[DCl['tp'][j][l],0,DCl['pp'][j][l]]]))

    #Initializing the Fisher matrix.
    F = np.zeros((npar, npar))
    f_sky = noise_exp()[2]
    #Computing the Fisher matrix elements.
    print(len(invCl_list),len(DCl_list[6]))
    for i in range(npar):
        for j in range(npar):
            for l in range(l_min,l_max+1):
                F[i][j] = F[i][j] + (2*l+1)/2*f_sky*np.trace(np.dot(invCl_list[l], np.dot(DCl_list[i][l], np.dot(invCl_list[l],DCl_list[j][l]))))

    return F    #Now the Fisher matrix contains the noise.