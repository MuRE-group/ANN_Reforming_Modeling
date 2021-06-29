import numpy  as np
import pandas as pd

import pickle

from scipy.integrate import odeint
from scipy.integrate import solve_ivp

import matplotlib
import matplotlib.pyplot as plt

np.random.seed(10)

#Function to compute equilibrium constant
def compute_K(vi, Ai ,Bi, Ci, Di, Gi, Hi, T_K):
    #Inputs:
    #       - vi: Stoichiometric vector of the given reaction
    #       - Ai, Bi, Ci, Di: Empirical values
    #       - H_i: Vector of enthalpies,          kJ/mol
    #       - G_i: Vector of Gibbs free energies, kJ/mol
    #Output:
    #       - K : Value of the equilibrium constant

    T0_K = 298.15; DIV_K = T_K/T0_K;

    A = np.dot(vi, Ai); B = np.dot(vi, Bi); C = np.dot(vi, Ci);
    G = np.dot(vi, Gi); H = np.dot(vi, Hi); D = np.dot(vi, Di);

    K0 = np.exp(-G * 1000 / (8.314 * T0_K));
    K1 = np.exp((H * 1000 / (8.314 * T0_K)) * (1 - T0_K / T_K));
    K2 = np.exp(A * (np.log(DIV_K) - (DIV_K - 1) / DIV_K) + 0.5 * B * T0_K * (DIV_K - 1) ** 2 / DIV_K
             + (1/6) * C * T0_K ** 2 * (DIV_K-1) ** 2 * (DIV_K + 2) / DIV_K
             + 0.5 * D * (DIV_K - 1) ** 2 / (T0_K * DIV_K) ** 2);

    K = K0 * K1 * K2;
    return K

#Conservation equation
def conservation_eq(F_v, tau, k_v, T_C, FN2, model):
    #Inputs:
    #       - tau: Space time,           gcat min molNaphtha-1
    #       - F_v: Vector of flow rates, Dimensionless, Fi/FNaphtha0
    #       - k_v: Vector of kinetics,   c.u.
    #       - T_C: Temperature,          C
    #       - FN2: Flow rate of N2,      Dimensionless, FN2/FNaphtha0
    #Outputs:
    #       - Solved mass balances of dimensionless flow rates

    T_K = T_C + 273.15;

    # WGS reaction (CO + H2O <=> H2 + CO2)
    v_WGS = np.array([-1, -1, 1, 1])
    A_WGS = np.array([3.376, 3.47, 3.249, 5.457])
    B_WGS = np.array([.557, 1.45, .422, 1.045])*1e-3;
    C_WGS = np.array([0, 0, 0, 0])*1e-6;
    D_WGS = np.array([-.031, .121, .083, -1.157])*1e5;
    G_WGS = np.array([-137.4, -228.8, 0, -394.6]);       #KJ/mol
    H_WGS = np.array([-110.525, -241.818, 0, -393.509]); #KJ/mol

    # SRM Reaction (CH4 + H2O <=> 3H2 + CO)
    v_SRM = np.array([-1, -1, 3, 1]);
    A_SRM = np.array([1.702, 3.47, 3.249, 3.376]);
    B_SRM = np.array([9.081, 1.45, .422, .557])*1e-3;
    C_SRM = np.array([-2.164, 0, 0, 0])*1e-6;
    D_SRM = np.array([0, .121, .083, -.031])*1e5;
    G_SRM = np.array([-50.46, -228.8, 0, -137.4]);       #KJ/mol
    H_SRM = np.array([-74.52, -241.818, 0, -110.525]);   #KJ/mol

    #Compute equilibrium constants for WGS, SRM and DRM
    K_WGS = compute_K(v_WGS, A_WGS, B_WGS, C_WGS, D_WGS, G_WGS, H_WGS, T_K);
    K_SRM = compute_K(v_SRM, A_SRM, B_SRM, C_SRM, D_SRM, G_SRM, H_SRM, T_K);
    FT = np.sum(F_v) + FN2; p = F_v / FT;

    #Rate constants from Arrhenius
    k_SRN = k_v[0] * np.exp((-k_v[1] / 8.31446) *(1 / T_K));
    k_WGS = k_v[2] * np.exp((-k_v[3] / 8.31446) *(1 / T_K));
    k_SRM = k_v[4] * np.exp((-k_v[5] / 8.31446) *(1 / T_K));

    #Adsorption constants
    K_N   = k_v[6] * np.exp(k_v[7] / (8.31446 * T_K));
    K_H2O = k_v[8] * np.exp(k_v[9] / (8.31446 * T_K));

    #Experimental power coefficients
    a = k_v[10]
    b = k_v[11]

    #Reaction rates
    if model == 'LH, molecular adsorption, different site':
        r_SRN = k_SRN * K_N * K_H2O * p[0] * p[1] / (1 + K_N * p[0]) / (1 + K_H2O * p[1]);

    elif model == 'LH, molecular adsorption, same site':
        r_SRN = k_SRN * K_N * K_H2O * p[0] * p[1] / ((1 + K_N * p[0] + K_H2O * p[1]) ** 2);

    elif model == 'LH, dissociative adsorption, different site':
        r_SRN = k_SRN * K_N * K_H2O * p[0] * p[1] / ((1 + K_N * p[0] * p[5] / p[1] + K_H2O * p[1] / p[5]) ** 2);

    elif model == 'LH, dissociative adsorption, same site':
        r_SRN = k_SRN * K_N * K_H2O * p[0] * p[1] / ((1 + np.sqrt(np.maximum((K_N * p[0]), 0)) + np.sqrt(np.maximum((K_H2O * p[1]), 0))) ** 2);

    elif model == 'ER, associative':
        r_SRN = k_SRN * K_N * p[0] * p[1] / (1 + K_N * p[0]);

    elif model == 'ER, dissociative':
        r_SRN = k_SRN * K_N * p[0] * p[1] / (1 + np.sqrt(np.maximum((K_N * p[0]), 0)));

    elif model == 'LH, dissociative (HC) and molecular (H2O), same site':
        r_SRN = k_SRN * K_N * K_H2O * p[0] * p[1] / ((1 + np.sqrt(np.maximum((K_N * p[0]), 0)) + K_H2O * p[1]) ** 2);

    elif model == 'Power Law':
        r_SRN = k_SRN * (np.maximum((p[0]), 0) ** a) * (np.maximum((p[1]), 0) ** b);

    r_WGS = k_WGS * (p[3] * p[1] - p[5] * p[2] / K_WGS);
    r_SRM = k_SRM * (p[4] * p[1] - (p[5] ** 3) * p[3] / K_SRM);

    #ODEs
    s_m = np.array([[-1, 0, 0], [-6.7, -1, -1], [0, 1, 0], [6.7, -1, 1], [0, 0, -1], [6.7 + 7.7, 1, 3]])
    r_m = np.array([r_SRN, r_WGS, r_SRM]);
    r_i = np.dot(s_m, r_m);

    return r_i

#Functon to run a single set of ODEs, for a given tauspan, initial conditions, temperature and set of k values
def run_ODE(tauspan, F0_v, T_K, k_v, model, ivp = False):

    F0N2 = F0_v[-1]
    F0_v = F0_v[:-1]
    T_C  = T_K - 273.15
    if ivp == True:
        res = solve_ivp(lambda tau, F_v: conservation_eq(F_v, tau, k_v, T_C, F0N2, model),
                    [tauspan[0], tauspan[-1]], F0_v, t_eval = tauspan, method = 'RK45'); res = res.y.T;
    else:
        args = (k_v, T_C, F0N2, model)
        res = odeint(conservation_eq, F0_v, tauspan, args, mxstep = 50000)
    return tauspan, res

global ne, nl, df_exp, df0_exp

#Read data from experimental excel
df_exp  = pd.read_excel('Raw_Data.xlsx')
#Extract initial conditions that need simulating
df0_exp = df_exp[df_exp[df_exp.columns[0]] == 0]
#Calculate number of experiments to perform simulation
ne  = len(df_exp[df_exp[df_exp.columns[0]] == 0])
#Calculate number of points at which ODEs should be solved
nl  = len(df_exp[df_exp.columns[0]].unique())

def multiple_ODEs(k_v, model, ivp = False):

    F = []

    for j in range(ne):
        index_no   = df0_exp.iloc[j].name
        T_K        = df0_exp[df0_exp.columns[1]].iloc[j]
        F0_v       = df0_exp[df0_exp.columns[2:9]].iloc[j].values
        F0N2       = df0_exp[df0_exp.columns[8]].iloc[j]
        df_j = df_exp.iloc[j * nl : nl * (j + 1),:]
        tauspan = df_j[(df_j[df_exp.columns[2]] > -0.1) & (df_j[df_exp.columns[2]] < 1.1)][df_exp.columns[0]].values
        tau, F_sol = run_ODE(tauspan, F0_v, T_K, k_v, model, ivp)

        F.append(F_sol)

    F = np.concatenate(F).ravel()

    return F

def pick_params(params_dict, instances_per_model, distribution = False):

    params = np.empty((len(params_dict), instances_per_model))

    i = 0

    for parameter in params_dict:
        params[i,:] = np.random.uniform(params_dict[parameter][0], params_dict[parameter][1], instances_per_model)
        i += 1


    plt.figure(figsize = (14,8))

    i = 0

    for parameter in params_dict:
        plt.subplot(3,4,i+1)
        count, bins, ignored = plt.hist(params[i,:], 10)
        plt.plot(bins, np.ones_like(bins) * np.mean(count), linewidth = 2, color = 'r')
        plt.title('{}'.format(parameter))
        plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
        plt.locator_params(axis = "x", nbins = 6)
        plt.tight_layout()
        i +=1
        plt.savefig('Params_distribution.png')

    if distribution == True:
        plt.show()

    plt.close()

    return params

def perform_model(model, params, sigmar, sigmac, instances_per_model):

    sol = []

    print('\nFollowing model: {}'.format(model))

    for i in range(instances_per_model):

        print('Attempting {} instance...'.format(i + 1))

        k_v = params[:,i]

        try:
            F = multiple_ODEs(k_v, model)
            epsilon = np.random.multivariate_normal(np.zeros(F.shape[0]), (np.identity(F.shape[0]) * (sigmar ** 2 * F / 100 + sigmac ** 2)))

        except:
            print('Exception ivp!\n')
            F = multiple_ODEs(k_v, model, ivp = True)
            epsilon = np.random.multivariate_normal(np.zeros(F.shape[0]), (np.identity(F.shape[0]) * (sigmar ** 2 * F / 100 + sigmac ** 2)))

        sol.append(F + epsilon)

    sol = np.asarray(sol)
    df = pd.DataFrame(sol)
    df['Label'] = model

    return df

def in_silico(models, params_dict, instances_per_model, sigmar, sigmac, distribution):

    Data = []

    print('Parameters sampled!')
    params = pick_params(params_dict, instances_per_model, distribution)
    pd.DataFrame(params.T, columns = list(params_dict.keys())).rename_axis('Samples').round(3).to_excel('Params_sampled.xlsx')

    for model in models:
        df_model = perform_model(model, params, sigmar, sigmac, instances_per_model)
        Data.append(df_model)
    print('\nDone!')

    Data = pd.concat(Data, axis = 0).reset_index()
    Data = Data.drop(Data.columns[0], axis = 1)
    return Data

params_dict = {'k0_SNR' : np.array([1.0E+07, 1.0E+08]),
               'Ea_SNR' : np.array([6.0E+04, 9.0E+04]),
               'k0_WGS' : np.array([2.0E+05, 3.0E+05]),
               'Ea_WGS' : np.array([4.0E+04, 7.0E+04]),
               'k0_SMR' : np.array([2.1E+11, 2.2E+11]),
               'Ea_SMR' : np.array([1.2E+05, 1.5E+05]),
               'K0_A'   : np.array([1.0E-02, 5.0E-02]),
               'AH_A'   : np.array([1.0E+04, 3.0E+04]),
               'K0_B'   : np.array([1.0E-03, 1.0E-02]),
               'AH_B'   : np.array([3.0E+04, 5.0E+04]),
                'a'     : np.array([0.25, 3]),
                'b'     : np.array([0.25, 3])}

instances_per_model = 500

models =  ['LH, molecular adsorption, different site',
           'LH, molecular adsorption, same site',
           'LH, dissociative adsorption, different site',
           'LH, dissociative adsorption, same site',
           'ER, associative',
           'ER, dissociative',
           'LH, dissociative (HC) and molecular (H2O), same site',
           'Power Law']

sigmar = 0.0
sigmac = 0.0

distribution = False

text_file = open('README_In_Silico.txt', 'a')
with open('README_In_Silico.txt','w') as file:
    file.write('Kinetic parameters: \n')
    file.write('\n{''\n')
    for k in sorted (params_dict.keys()):
        file.write("'%s':'%s', \n" % (k, params_dict[k]))
    file.write('}\n')
    file.write('\nInstances per model =  %s\n' % instances_per_model)
    file.write('\nModels: \n')
    file.write('\n')
    for k in models:
        file.write("'%s'\n" % (k))
    file.write('\nNoise parameters:  \n')
    file.write('\nSigmaR =  %s\n' % sigmar)
    file.write('SigmaC =  %s\n' % sigmac)

Data = in_silico(models, params_dict, instances_per_model, sigmar, sigmac, distribution)

# Save Data for Naphtha Reforming in csv and xlsx format
Data.to_csv('Data_in_silico' + '_' + str(instances_per_model) + '.csv')
Data.to_excel('Data_in_silico' + '_' + str(instances_per_model) + '.xlsx')

#Save the list with models tested
filename = 'model_list' + '_' + str(instances_per_model) + '.sav'
outfile = open(filename,'wb')
pickle.dump(models, outfile)
outfile.close()
