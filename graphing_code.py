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
p_s = 3.07
p_v = 8.52
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

Vs0 = 0.31
Vv0 = 0.0

initial_conditions = [T0, Es0, Is0, Ev0, Iv0, Vs0, Vv0]

Vv0_values = 0.148*np.logspace(0,6,100)
beta_values = 4.71e-8*np.logspace(-2,4,100)

# Arrays to store max values and durations
max_Vs_values = np.zeros((len(beta_values), len(Vv0_values)))
max_Vv_values = np.zeros((len(beta_values), len(Vv0_values)))
time_above_threshold_Vs_values = np.zeros((len(beta_values), len(Vv0_values)))
time_above_threshold_Vv_values = np.zeros((len(beta_values), len(Vv0_values)))

max_Vs_values = np.loadtxt("Smax.dat")
time_above_threshold_Vs_values = np.loadtxt("Stime.dat")
max_Vv_values = np.loadtxt("Vmax.dat")
time_above_threshold_Vv_values = np.loadtxt("Vtime.dat")
AUC_Vs = np.loadtxt("SAUC.dat")
AUC_Vv = np.loadtxt("VAUC.dat")

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
plt.savefig("stochastic-2.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()
