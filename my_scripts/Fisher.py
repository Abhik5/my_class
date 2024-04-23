import numpy as np
import matplotlib.pyplot as plt
import math
from classy import Class 

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
    'l_max_scalars': 2500,
    'lensing': 'yes',
    'omega_b': omega_b, 
    'omega_cdm': omega_cdm,
    'h': h,
    'A_s':2.100549e-09,
    'n_s':0.9660499,
    'tau_reio':0.05430842,
    'N_eff':N_ur
}
print("omega_b=",omega_b,"omega_cdm=",omega_cdm,"h=",h,"N_eff=",N_ur)
"""
file_path = 'input_dict.ini'        #Contains the parameters used in the preliminary CLASS run
def create_dictionary_from_file(file_path):
    
    #Create a dictionary from data in a file where each line contains key-value pairs.

    #Parameters:
        #file_path (str): Path to the file containing key-value pairs.

    #Returns:
        #dict: Dictionary containing the key-value pairs.
    
    # Initialize an empty dictionary
    my_dict = {}

    # Read data from the file
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line by '=' to separate key and value
            key, value = line.strip().split('=')
            # Strip any leading/trailing spaces from key and value
            key = key.strip()
            value = value.strip()
            # Add key-value pair to the dictionary
            my_dict[key] = value

    return my_dict


params = create_dictionary_from_file(file_path)

file_path = 'nonthermal04_cl_lensed.dat'
def read_cl_values(file_path):
    
    # Read the C_l values from the file
    temp_cl = np.loadtxt(file_path)

    # Extract ell and C_l values

    return temp_cl

#Arrays containing the l's and C_l's
ell = read_cl_values(file_path)[:,0]
ClTT = read_cl_values(file_path)[:,1]
ClEE = read_cl_values(file_path)[:,2]
ClTE = read_cl_values(file_path)[:,3]
ClPP = read_cl_values(file_path)[:,5]
ClTP = read_cl_values(file_path)[:,6]

#Function to compute the derivatives
def get_deriv(params, theta, ch, theta_step):
    channel = {'tt':1,'ee':2,'te':3,'pp':5,'tp':6}
    i = channel[ch]
    left_params = params.copy()
    left_params[theta] = params[theta] - theta_step
    right_params = params.copy()
    right_params[theta] = params[theta] + theta_step
    cl_left = read_cl_values(left_params)[:,i]
    cl_right = read_cl_values(right_params)[:,i]
    dCl_dtheta = (cl_right - cl_left) / (2 * theta_step)
    return dCl_dtheta
"""

def utility_function_call_CLASS(input_dict, l_max=2500):
    #Compute Cl with this utility function, repeat less code.
    cosmo = Class()
    cosmo.set(input_dict)
    cosmo.compute()
    temp_cl = cosmo.lensed_cl(l_max)
    cosmo.struct_cleanup()
    cosmo.empty()
    return temp_cl

ClTT = utility_function_call_CLASS(params)['tt']
ClTP = utility_function_call_CLASS(params)['tp']

print("ClTT[2000]=",ClTT[2000])
print("ClTP[1500]=",ClTP[1500])
"""
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

#Arrays containing the derivatives of C_l's w.r.t all 7 parameters
key_array = ['omega_b','omega_cdm','h','A_s','n_s','tau_reio','N_eff']
step_array = [0.01,0.01,0.01,1e-10,0.1,0.01,0.01]
D_ClTT = [[] for _ in range(7)]
D_ClTE = [[] for _ in range(7)]
D_ClEE = [[] for _ in range(7)]
D_ClPP = [[] for _ in range(7)]
D_ClTP = [[] for _ in range(7)]

for i in range(7):
    D_ClTT[i] = get_deriv( params,key_array[i],'tt',step_array[i])
    D_ClTE[i] = get_deriv( params,key_array[i],'te',step_array[i])
    D_ClEE[i] = get_deriv( params,key_array[i],'ee',step_array[i])
    D_ClPP[i] = get_deriv( params,key_array[i],'pp',step_array[i])
    D_ClTP[i] = get_deriv( params,key_array[i],'tp',step_array[i])

DCl = [[] for _ in range(7)]
Cl_list = []
invC = []
l_min = 2
l_max = 2500
for l in range(l_min,l_max+1):
    Cl = np.array([[ClTT[l],ClTE[l],ClTP[l]],[ClTE[l],ClEE[l],0],[ClTP[l],0,ClPP[l]]])
    if np.linalg.det(Cl)==0:
        print("C_%d",l," not invertible.")
        break
    else:
        Cl_list.append(Cl)
        for j in range(7):
            DCl[j].append(np.array([[D_ClTT[j][l],D_ClTE[j][l],D_ClTP[j][l]],[D_ClTE[j][l],D_ClEE[j][l],0],[D_ClPP[j][l],0,D_ClPP[j][l]]]))
        Cl_inverse = np.linalg.inv(Cl)
        invC.append(Cl_inverse)
l_max = len(invC)
invC_list = np.array(invC)
print("C_1000=",Cl_list[1000])
#Arrays of matrices indexed by the value of l.

#This is the array of lists of the derivatives
#DC_list = [D0C_list,D1C_list,D2C_list]
DC_list = [DCl[0],DCl[1],DCl[2],DCl[3],DCl[4],DCl[5],DCl[6]]

#Computing the Fisher matrix elements.
f_sky = 0.4  #0.4 for CMB-S4

#Noise parameters
sT = 33 * (np.pi/60./180.)
sP = sT * np.sqrt(2.)
theta_FWHM = 7. * (np.pi/60./180.)

NlTT = []
NlEE = []
NlBB = []
NlPP = []
for l in range(2500):
    NlTT.append(sT**2*math.exp(l*(l+1)*theta_FWHM**2/(8*math.log(2))))
    NlEE.append(sP**2*math.exp(l*(l+1)*theta_FWHM**2/(8*math.log(2))))
    NlBB.append(sP**2*math.exp(l*(l+1)*theta_FWHM**2/(8*math.log(2))))
#Initializing the Fisher matrix.
F2 = [[0, 0, 0, 0, 0, 0, 0] for _ in range(7)]
#Computing the Fisher matrix elements.
for i in range(7):
    for j in range(7):
        for l in range(l_min,l_max):
            F2[i][j] = F2[i][j] + (2*l+1)/2*f_sky*np.trace(np.dot(invC_list[l], np.dot(DC_list[i][l], np.dot(invC_list[l],DC_list[j][l]))))

print("F =",F2)      #Now the Fisher matrix contains the noise.
#The new covariance matrix
Cov = np.linalg.inv(F2)
print("Cov =",Cov)

#Now we can use this Cov matrix in the Fisher analysis code in jupyter notebook
#===================================================================================================================================================================="""
