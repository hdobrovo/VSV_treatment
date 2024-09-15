import numpy as np
import matplotlib.pyplot as plt
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

Vs0 = 1 #0.31
Vv0 = 0.0

initial_conditions = [T0, Es0, Is0, Ev0, Iv0, Vs0, Vv0]

Vv0_values = np.logspace(-1,5,100)
beta_values = 4.71e-8*0.31*np.logspace(-2,4,100)

# Arrays to store max values and durations
max_Vs_values = np.zeros((len(beta_values), len(Vv0_values)))
max_Vv_values = np.zeros((len(beta_values), len(Vv0_values)))
time_above_threshold_Vs_values = np.zeros((len(beta_values), len(Vv0_values)))
time_above_threshold_Vv_values = np.zeros((len(beta_values), len(Vv0_values)))
AUC_Vs = np.zeros((len(beta_values), len(Vv0_values)))
AUC_Vv = np.zeros((len(beta_values), len(Vv0_values)))

for i, beta_val in enumerate(beta_values):
	for j, Vv0_val in enumerate(Vv0_values):
		#Delay arrays
		delay_values = [1]

		#Empty array for results
		results = []


		for delay in delay_values:
			# Time points
			times = np.linspace(0, delay, delay*100)
			sol_initial = odeint(viral_infection, initial_conditions, times, 
			args=(beta_val, beta_val, k_s, k_v, delta_s, delta_v, p_s, p_v, c_s, c_v))

			T = sol_initial[:, 0]
			Es = sol_initial[:, 1]
			Is = sol_initial[:, 2]
			Ev = sol_initial[:, 3]
			Iv = sol_initial[:, 4]
			Vs = sol_initial[:, 5]
			Vv = sol_initial[:, 6]

			#Adding Vv and setting initial conditions for second ODEINT
			times_extended = np.linspace(delay, 50, (50-delay)*100)

			initial_conditions2 = [T[-1], Es[-1], Is[-1], Ev[-1], Iv[-1], Vs[-1], Vv0_val]

			# Solve the viral infection model for the second part (1 day to 30 days)
			sol_extended = odeint(viral_infection, initial_conditions2, times_extended, 
			args=(beta_val, beta_val, k_s, k_v, delta_s, delta_v, p_s, p_v, c_s, c_v))

			# Combine the results of the two runs by removing the last point from the initial run
			T_combined = np.concatenate((sol_initial[:-1, 0], sol_extended[:, 0]))
			Es_combined = np.concatenate((sol_initial[:-1, 1], sol_extended[:, 1]))
			Is_combined = np.concatenate((sol_initial[:-1, 2], sol_extended[:, 2]))
			Ev_combined = np.concatenate((sol_initial[:-1, 3], sol_extended[:, 3]))
			Iv_combined = np.concatenate((sol_initial[:-1, 4], sol_extended[:, 4]))
			Vs_combined = np.concatenate((sol_initial[:-1, 5], sol_extended[:, 5]))
			Vv_combined = np.concatenate((sol_initial[:-1, 6], sol_extended[:, 6]))

			#Combining times
			times_combined = np.concatenate((times[:-1], times_extended))
			#plt.plot(times_combined,np.log10(Vs_combined),'r')
			#plt.plot(times_combined,np.log10(Vv_combined),'b')
			#plt.show()
			#Finding max
			max_Vs = np.max(Vs_combined)
			max_Vv = np.max(Vv_combined)
			max_Vs_values[i, j] = max_Vs
			max_Vv_values[i, j] = max_Vv
			AUC_Vs[i,j] = np.trapz(Vs_combined)
			AUC_Vv[i,j] = np.trapz(Vv_combined)
			print([max_Vs,max_Vv])
			#Time duration Vs is above threshold
			threshold = 10**1
			crossed_threshold_indices_Vs = np.where(Vs_combined > threshold)[0]

			if len(crossed_threshold_indices_Vs) > 0:
				first_crossed_index_Vs = crossed_threshold_indices_Vs[0]
				last_crossed_index_Vs = crossed_threshold_indices_Vs[-1]

				time_above_threshold_Vs = times_combined[first_crossed_index_Vs:last_crossed_index_Vs+1]
				duration_above_threshold_Vs = time_above_threshold_Vs[-1] - time_above_threshold_Vs[0]
				time_above_threshold_Vs_values[i, j] = duration_above_threshold_Vs


			crossed_threshold_indices_Vv = np.where(Vv_combined > threshold)[0]

			if len(crossed_threshold_indices_Vv) > 0:
				first_crossed_index_Vv = crossed_threshold_indices_Vv[0]
				last_crossed_index_Vv = crossed_threshold_indices_Vv[-1]
				time_above_threshold_Vv = times_combined[first_crossed_index_Vv:last_crossed_index_Vv+1]
				duration_above_threshold_Vv = time_above_threshold_Vv[-1] - time_above_threshold_Vv[0]
				time_above_threshold_Vv_values[i, j] = duration_above_threshold_Vv


# Create the heatmaps

plt.figure(figsize=(24, 12))

# Heatmap for Maximum Vs vs. Gamma and Vv0
plt.subplot(2, 3, 1)
plt.imshow(np.log10(max_Vs_values), cmap='hot', extent=[-1,5,-1,5], aspect='auto', origin='lower')
plt.colorbar()
plt.text(5.1, 0.5, r'Peak SARS-CoV-2', fontsize=18, rotation=90)
plt.ylabel(r'Infection rate (/d)',fontsize=22)
plt.xlabel(r'Initial VSV dose',fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.yticks( np.arange(-1,6,1), ('$10^{-10}$','$10^{-9}$','$10^{-8}$','$10^{-7}$','$10^{-6}$','$10^{-5}$','$10^{-4}$'), fontsize=22 )
plt.xticks( np.arange(-1,6,1), ('$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$'), fontsize=22 )
#plt.title('Maximum Vs vs. Gamma and Vv0')

# Heatmap for Time above threshold Vs vs. Gamma and Vv0
plt.subplot(2, 3, 2)
plt.imshow((time_above_threshold_Vs_values), cmap='hot', extent=[-1,5,-1,5], aspect='auto', origin='lower', vmin = 0, vmax = 50)
plt.colorbar()
plt.text(5.1, 0, r'SARS-CoV-2 duration', fontsize=18, rotation=90)
plt.ylabel(r'Infection rate (/d)',fontsize=22)
plt.xlabel(r'Initial VSV dose',fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.yticks( np.arange(-1,6,1), ('$10^{-10}$','$10^{-9}$','$10^{-8}$','$10^{-7}$','$10^{-6}$','$10^{-5}$','$10^{-4}$'), fontsize=22 )
plt.xticks( np.arange(-1,6,1), ('$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$'), fontsize=22 )
#plt.title('Maximum Vs vs. Gamma and Vv0')

# Heatmap for AUC Vs vs. Gamma and Vv0
plt.subplot(2, 3, 3)
plt.imshow(np.log10(AUC_Vs), cmap='hot', extent=[-1,5,-1,5], aspect='auto', origin='lower')
plt.colorbar()
plt.text(5.1, 0, r'SARS-CoV-2 AUC', fontsize=18, rotation=90)
plt.ylabel(r'Infection rate (/d)',fontsize=22)
plt.xlabel(r'Initial VSV dose',fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.yticks( np.arange(-1,6,1), ('$10^{-10}$','$10^{-9}$','$10^{-8}$','$10^{-7}$','$10^{-6}$','$10^{-5}$','$10^{-4}$'), fontsize=22 )
plt.xticks( np.arange(-1,6,1), ('$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$'), fontsize=22 )
#plt.title('Maximum Vs vs. Gamma and Vv0')

# Heatmap for Maximum Vv vs. Gamma and Vv0
plt.subplot(2, 3, 4)
plt.imshow(np.log10(max_Vv_values), cmap='hot', extent=[-1,5,-1,5], aspect='auto', origin='lower')
plt.colorbar()
plt.text(5.1, 1, r'Peak VSV', fontsize=18, rotation=90)
plt.ylabel(r'Infection rate (/d)',fontsize=22)
plt.xlabel(r'Initial VSV dose',fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.yticks( np.arange(-1,6,1), ('$10^{-10}$','$10^{-9}$','$10^{-8}$','$10^{-7}$','$10^{-6}$','$10^{-5}$','$10^{-4}$'), fontsize=22 )
plt.xticks( np.arange(-1,6,1), ('$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$'), fontsize=22 )
#plt.title('Maximum Vs vs. Gamma and Vv0')

# Heatmap for Time above threshold Vv vs. Gamma and Vv0
plt.subplot(2, 3, 5)
plt.imshow((time_above_threshold_Vv_values), cmap='hot', extent=[-1,5,-1,5], aspect='auto', origin='lower', vmin=0, vmax=50)
plt.colorbar()
plt.text(5.1, 0.5, r'VSV duration', fontsize=18, rotation=90)
plt.ylabel(r'Infection rate (/d)',fontsize=22)
plt.xlabel(r'Initial VSV dose',fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.yticks( np.arange(-1,6,1), ('$10^{-10}$','$10^{-9}$','$10^{-8}$','$10^{-7}$','$10^{-6}$','$10^{-5}$','$10^{-4}$'), fontsize=22 )
plt.xticks( np.arange(-1,6,1), ('$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$'), fontsize=22 )
#plt.title('Maximum Vs vs. Gamma and Vv0')

# Heatmap for AUC Vs vs. Gamma and Vv0
plt.subplot(2, 3, 6)
plt.imshow(np.log10(AUC_Vv), cmap='hot', extent=[-1,5,-1,5], aspect='auto', origin='lower')
plt.colorbar()
plt.text(5.1, 0.7, r'VSV AUC', fontsize=18, rotation=90)
plt.ylabel(r'Infection rate (/d)',fontsize=22)
plt.xlabel(r'Initial VSV dose',fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.yticks( np.arange(-1,6,1), ('$10^{-10}$','$10^{-9}$','$10^{-8}$','$10^{-7}$','$10^{-6}$','$10^{-5}$','$10^{-4}$'), fontsize=22 )
plt.xticks( np.arange(-1,6,1), ('$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$'), fontsize=22 )
#plt.title('Maximum Vs vs. Gamma and Vv0')

plt.tight_layout()
plt.savefig("treatment1d-2.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()
