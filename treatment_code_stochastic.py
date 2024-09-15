import numpy as np
from scipy.integrate import odeint

def viral_infection(variables, t, beta_s, beta_v, k_s, k_v, delta_s, delta_v, p_s, p_v, c_s, c_v):
    T, Es, Is, Ev, Iv, Vs, Vv = variables

    dT_dt = -beta_s * T * Vs
    dEs_dt = beta_s * T * Vs - k_s * Es
    dIs_dt = k_s * Es - delta_s * Is - beta_v * Is * Vv
    dEv_dt = beta_v * Is * Vv - k_v * Ev
    dIv_dt = k_v * Ev - delta_v * Iv
    dVs_dt = p_s * (Is + Ev + Iv) - c_s * Vs - beta_v * Vv * Vs
    dVv_dt = p_v * Iv - c_v * Vv 

    return [dT_dt, dEs_dt, dIs_dt, dEv_dt, dIv_dt, dVs_dt, dVv_dt]
    

def stoc_eqs_tauleap(INP):
    X = INP
    Rate = np.zeros((11))##propensity function [b1*T*V1 k1*E1 d1*I1 p1*I1 c1*V1 b2*T*V2 k2*E2 d2*I2 p2*I2 c2*V2]
    Transitions = np.zeros((11,7))##stoichiometric matrix, each row of which is a transition vector
    Rate[0] = beta*X[0]*X[5]; Transitions[0,:]=([-1, +1, 0, 0, 0, 0, 0])
    Rate[1] = k_s*X[1];  Transitions[1,:]=([0, -1, +1, 0, 0, 0, 0])
    Rate[2] = delta_s*X[2];  Transitions[2,:]=([0, 0, -1, 0, 0, 0, 0])
    Rate[3] = beta*X[2]*X[6];  Transitions[3,:]=([0, 0, -1, +1, 0, 0, 0])
    Rate[4] = k_v*X[3];  Transitions[4,:]=([0, 0, 0, -1, +1, 0, 0])
    Rate[5] = delta_v*X[4]; Transitions[5,:]=([0, 0, 0, 0, -1, 0, 0])
    Rate[6] = p_s*(X[2]+X[3]+X[4]);  Transitions[6,:]=([0, 0, 0, 0, 0, 1, 0])
    Rate[7] = c_s*X[5];  Transitions[7,:]=([0, 0, 0, 0, 0, -1, 0])
    Rate[8] = p_v*X[4];  Transitions[8,:]=([0, 0, 0, 0, 0, 0, +1])
    Rate[9] = c_v*X[6];  Transitions[9,:]=([0, 0, 0, 0, 0, 0, -1])
    Rate[10] = beta_v*X[5]*X[6]; Transitions[10,:]=([0, 0, 0, 0, 0, -1, 0])    
    for k in range(11):
    	leap=np.random.poisson(Rate[k]*tau);#no of times each transition occurs in the time interval dt or tau
    		## To avoid negative numbers
    	Use=min([leap, X[np.where(Transitions[k,:]<0)]]);
    	X=X+Transitions[k,:]*Use;
    return X

def Stoch_Iteration(INPUT):
    tt=0
    T = [0] 
    Es = [0]
    Is = [0]
    Ev = [0]
    Iv = [0]
    Vs = [0]
    Vv = [0]
    for tt in time:
        res=stoc_eqs_tauleap(INPUT)
        T.append(INPUT[0])
        Es.append(INPUT[1])
        Is.append(INPUT[2])
        Ev.append(INPUT[3])
        Iv.append(INPUT[4])
        Vs.append(INPUT[5])
        Vv.append(INPUT[6])
        INPUT=res
    return ([T, Es, Is, Ev, Iv, Vs, Vv])
    
# Parameter values
beta_s = 4.71e-8
beta_v = 4.71e-8
p_s = 3.07/0.31
p_v = 8.52/0.148
k_s = 5.0
k_v = 487
delta_s = 1.07
delta_v = 1.33
c_s = 2.4
c_v = 1.33



# Initial conditions
T0 = 4e8

Es0 = 0
Ev0 = 0
Is0 = 0
Iv0 = 0

Vs0 = 1
Vv0 = 0.0

#initial_conditions = [T0, Es0, Is0, Ev0, Iv0, Vs0, Vv0]
tau = 0.001 
Max = 50
INPUT = np.array((T0, Es0, Is0, Ev0, Iv0, Vs0, Vv0))

Vv0_values = np.logspace(0,6,100)
beta_values = 4.71e-8*0.31*np.logspace(-2,4,100)
n_simulations = 10

# Arrays to store max values and durations
max_Vs_values = np.zeros((len(beta_values), len(Vv0_values)))
max_Vv_values = np.zeros((len(beta_values), len(Vv0_values)))
time_above_threshold_Vs_values = np.zeros((len(beta_values), len(Vv0_values)))
time_above_threshold_Vv_values = np.zeros((len(beta_values), len(Vv0_values)))
AUC_Vs = np.zeros((len(beta_values), len(Vv0_values)))
AUC_Vv = np.zeros((len(beta_values), len(Vv0_values)))

for i, beta_val in enumerate(beta_values):
    for j, Vv0_val in enumerate(Vv0_values):
        max_Vs = 0
        max_Vv = 0
        duration_above_threshold_Vv = 0
        duration_above_threshold_Vs = 0
        for l in range(1, n_simulations+1):
            time=np.arange(0.0, Max, tau)
            beta = beta_val

            #Adding Vv and setting initial conditions for second ODEINT
            INPUT = np.array((T0, Es0, Is0, Ev0, Iv0, Vs0, Vv0_val))
            #initial_conditions2 = [T[-1], Es[-1], Is[-1], Ev[-1], Iv[-1], Vs[-1], Vv0_val]

            # Solve the viral infection model for the second part (1 day to 30 days)
            [T_combined, Es_combined, Is_combined, Ev_combined, Iv_combined, Vs_combined, Vv_combined] = Stoch_Iteration(INPUT)

            #plt.plot(np.log10(Vs_combined),'r')
            #plt.plot(np.log10(Vv_combined),'b')
            #plt.show()
            #Finding max
            max_Vs = max_Vs + np.max(Vs_combined)
            max_Vv = max_Vv + np.max(Vv_combined)
            AUC_Vs[i,j] = np.trapz(Vs_combined)
            AUC_Vv[i,j] = np.trapz(Vv_combined)          

            #Time duration Vs is above threshold
            threshold = 10**1
            crossed_threshold_indices_Vs = np.where(np.asarray(Vs_combined) > threshold)[0]

            if len(crossed_threshold_indices_Vs) > 0:
                first_crossed_index_Vs = crossed_threshold_indices_Vs[0]
                last_crossed_index_Vs = crossed_threshold_indices_Vs[-1]
                time_above_threshold_Vs = (last_crossed_index_Vs - first_crossed_index_Vs)*tau
                duration_above_threshold_Vs = duration_above_threshold_Vs + time_above_threshold_Vs
			     
            crossed_threshold_indices_Vv = np.where(np.asarray(Vv_combined) > threshold)[0]
            if len(crossed_threshold_indices_Vv) > 0:
                first_crossed_index_Vv = crossed_threshold_indices_Vv[0]
                last_crossed_index_Vv = crossed_threshold_indices_Vv[-1]
                time_above_threshold_Vv = (last_crossed_index_Vv - first_crossed_index_Vv)*tau
                duration_above_threshold_Vv = duration_above_threshold_Vv + time_above_threshold_Vv
			     
        max_Vs_values[i, j] = max_Vs/n_simulations
        max_Vv_values[i, j] = max_Vv/n_simulations
        time_above_threshold_Vs_values[i, j] = duration_above_threshold_Vs/n_simulations
        time_above_threshold_Vv_values[i, j] = duration_above_threshold_Vv/n_simulations

np.savetxt('Smax.dat',max_Vs_values)
np.savetxt('Vmax.dat',max_Vv_values)
np.savetxt('Stime.dat',time_above_threshold_Vs_values)
np.savetxt('Vtime.dat',time_above_threshold_Vv_values)
np.savetxt('SAUC.dat',AUC_Vs)
np.savetxt('VAUC.dat',AUC_Vv)
