# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:04:33 2022

@author: roger
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem

# definition of the geometry
l = 10      # length of th house in meters
b = 5       # width of the house in meters
h = 2.5     # height of a Room in meters
Sw1 = 2 * l * h  # Surface of the side Walls
Sw2 = b * h  # Surface of the Windows and opposite Wall
Sg = l * b   # Surface of the Roof and grounds
V1 = l * b * h  # Volume of one Room


# Thermo-physical properties

air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)

wall = {'Conductivity': [1.5, 1.4, 1.4, 1.4, 1.4, 1.4,
                         0.027, 0.027, 0.027, 1.4],  # W/(m·K)
        'Density': [1500, 2300, 2300, 2300, 2300, 2300,
                    55, 55, 55, 2500],  # kg/m³
        'Specific heat': [1000, 880, 880, 880, 880, 880,
                          1210, 1210, 1210, 750],  # J/(kg·K)
        'Width': [1, 0.5, 0.3, 0.2, 0.2, 0.2, 0.08, 0.08, 0.08, 0.004],
        'Surface': [Sg, Sg, Sg, Sg, Sw1, Sw2, Sw1, Sw2, Sg, Sw2],  # m²
        'Slices': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}       # number of  slices
wall = pd.DataFrame(wall, index=['ground', 'Concrete_ground', 'Concrete_mid',
                                 'Concrete_roof', 'Concrete1',
                                 'Concrete2', 'Insulation1',
                                 'Insulation2', 'Insulation_roof', 'Glass'])

# Optical parameters
ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete)
ε_wLWf = 0.885    # long wave emmisivity: floor surface (wood color)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass
σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant

# Viewfactor definition
F = np.zeros([4, 4])
F[0, 1] = F[0, 3] = 0.075  # Viewfactor glass-ground/ceiling
F[2, 3] = F[2, 1] = 0.055  # Viewfactor wall-ground/ceiling
F[0, 2] = 1 - F[0, 1] - F[0, 3]  # Viewfactor glass-wall
F[1, 3] = F[3, 1] = 0.52   # Viewfactor roof-ground

# Building the different Matrices
# Incidence Matrix A
brn = 39  # number of branches
nds = 22  # number of nodes

A = np.zeros([brn, nds])
A[0, 0] = 1                 # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1    # branch 2: node 1 -> node 2
A[3, 2], A[3, 3] = -1, 1    # branch 3: node 2 -> node 3
A[4, 3], A[4, 4] = -1, 1    # branch 4: node 3 -> node 4
A[5, 3], A[5, 5] = -1, 1    # branch 5: node 3 -> node 5
A[6, 4], A[6, 5] = -1, 1    # branch 6: node 4 -> node 5
A[7, 6] = 1                 # branch 7: -> node 6
A[8, 6], A[8, 4] = -1, 1    # branch 8: node 6 -> node 4
A[9, 7] = 1                 # branch 9: -> node 7
A[10, 7], A[10, 8] = -1, 1  # branch 10: node 7 -> node 8
A[11, 8], A[11, 9] = -1, 1  # branch 11: node 8 -> node 10
A[12, 9], A[12, 10] = -1, 1  # branch 12: node 10 -> node 11
A[13, 10], A[13, 11] = -1, 1  # branch 13: node 11 -> node 12
A[14, 3], A[14, 11] = -1, 1   # branch 14: node 3 -> node 11
A[15, 4], A[15, 11] = -1, 1   # branch 15: node 4 -> node 11
A[16, 11], A[16, 5] = -1, 1   # branch 16: node 11 -> node 5
A[17, 5] = 1                  # branch 17: -> node 5
A[18, 5] = 1                  # branch 18: -> node 5
A[19, 5], A[19, 12] = -1, 1   # branch 19: node 5 -> node 12
A[20, 12], A[20, 13] = -1, 1    # branch 20: node 12 -> node 13
A[21, 13], A[21, 21] = -1, 1    # branch 21: node 13 -> node 21
A[22, 14] = 1                   # branch 22: -> node 14
A[23, 14], A[23, 15] = -1, 1    # branch 23: node 14 -> node 15
A[24, 15], A[24, 16] = -1, 1    # branch 24: node 15 -> node 16
A[25, 16], A[25, 17] = -1, 1    # branch 25: node 16 -> node 17
A[26, 17], A[26, 19] = -1, 1    # branch 26: node 17 -> node 19
A[27, 17], A[27, 18] = -1, 1    # branch 27: node 17 -> node 18
A[28, 18], A[28, 19] = -1, 1    # branch 28: node 18 -> node 19
A[29, 20] = 1                   # branch 29: -> node 20
A[30, 20], A[30, 18] = -1, 1    # branch 30: node 20 -> node 18
A[31, 19] = 1                   # branch 31: -> node 19
A[32, 19] = 1                   # branch 32: -> node 19
A[33, 21], A[33, 19] = -1, 1    # branch 33: node 21 -> node 19
A[34, 17], A[34, 21] = -1, 1    # branch 34: node 17 -> node 21
A[35, 18], A[35, 21] = -1, 1    # branch 35: node 18 -> node 21
A[36, 12], A[36, 3] = -1, 1     # branch 36: node 12 -> node 3
A[37, 12], A[37, 4] = -1, 1     # branch 37: node 12 -> node 4
A[38, 12], A[38, 11] = -1, 1    # branch 38: node 12 -> node 11

# calculation of Conductances
# convection parameters for the air
conp = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])  # W/(m²⋅K)


# define the Thermal conductances
R_cd = wall['Width'] / (wall['Conductivity'] * wall['Surface'])

# define the convection formulas
Rgr = 1 / (conp.iloc[0]['in'] * wall['Surface'][1])     # concrete ground
Rmid1 = 1 / (conp.iloc[0]['in'] * wall['Surface'][2])  # concrete middle ceiling
Rmid2 = 1 / (conp.iloc[0]['in'] * wall['Surface'][2])   # concrete middle floor
Rroof = 1 / (conp * wall['Surface'][3])    # concrete middle roof/ground
Rw1 = 1 / (conp * wall['Surface'][4])     # wall on the side
Rw2 = 1 / (conp * wall['Surface'][5])     # wall on the opposide of the window
Rg = 1 / (conp * wall['Surface'][9])     # from the glass

# longwave radiation calculations
Tm = 20 + 273   # K, mean temp for radiative exchange

# glass view
# glass and floor
GLWF1 = 4 * σ * Tm**3 * ε_wLWf / (1 - ε_wLWf) * Sg
GLWF10 = 4 * σ * Tm**3 * F[0, 1] * Sg
GLWF0 = 4 * σ * Tm**3 * ε_gLW / (1 - ε_gLW) * wall['Surface']['Glass']
RLWGF = (1 / GLWF1 + 1 / GLWF10 + 1 / GLWF0)

# glass and ceiling
GLWC3 = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * Sg
GLWC30 = 4 * σ * Tm**3 * F[0, 3] * Sg
GLWC0 = GLWF0
RLWGC = (1 / GLWC3 + 1 / GLWC30 + 1 / GLWC0)

# glass and Wall/ wall and glass
GLW2 = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * Sg
GLW20 = 4 * σ * Tm**3 * F[0, 2] * Sg
GLW0 = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * wall['Surface']['Glass']
RLWGW = (1 / GLW2 + 1 / GLW20 + 1 / GLW0)

# Wall view
# wall and floor
GLWF12 = 4 * σ * Tm**3 * F[2, 1] * Sg
GLWF2 = 4 * σ * Tm**3 * ε_gLW / (1 - ε_gLW) * (Sw1 + Sw2)
RLWWF = (1 / GLWF1 + 1 / GLWF12 + 1 / GLWF2)

# wall and ceiling
GLWC32 = 4 * σ * Tm**3 * F[2, 3] * Sg
GLWC2 = GLWF2
RLWWC = (1 / GLWC3 + 1 / GLWC32 + 1 / GLWC2)

# floor view - ceiling view
# floor and ceiling
GLW31 = 4 * σ * Tm**3 * F[0, 2] * Sg
RLWFC = (1 / GLWC3 + 1 / GLW31 + 1 / GLWF1)


# implementing he advection part
Va = V1                     # m³ volume of air
ACH = 1                    # air changes per hour
Va_dot = ACH / 3600 * Va    # m³/s air infiltration

# ventilation & advection
Rv = air['Density'] * air['Specific heat'] * Va_dot

# controller gain
Kp = 500

# glass: convection outdoor & conduction
Rgs = float(Rg['out'] + (2 * R_cd['Glass']))


# Conductance Matrix G
# number of temperature nodes and flow branches

R = np.zeros(brn)
R[0] = Rw1.iloc[0]['out'] + Rw2.iloc[0]['out']
R[1] = 1 / 2 * (R_cd['Insulation1'] + R_cd['Insulation2'])
R[2] = 1 / 2 * (R_cd['Insulation1'] + R_cd['Insulation2'] +
                R_cd['Concrete1'] + R_cd['Concrete2'])
R[3] = 1 / 2 * (R_cd['Concrete1'] + R_cd['Concrete2'])
R[4] = RLWGW
R[5] = Rw1.iloc[0]['in'] + Rw2.iloc[0]['in']
R[6] = Rg.iloc[0]['in']
R[7] = Rgs
R[8] = 1 / 2 * R_cd['Glass']
R[9] = 1 / 2 * R_cd['ground']
R[10] = 1 / 4 * R_cd['ground']
R[11] = 1 / 4 * R_cd['ground']
R[12] = 1 / 2 * R_cd['ground'] + 2 * R_cd['Concrete_ground']
R[13] = 1 / 2 * R_cd['Concrete_ground']
R[14] = RLWWF
R[15] = RLWGF
R[16] = Rgr
R[17] = 1
R[18] = Rv
R[19] = Rmid1
R[20] = 1 / 2 * R_cd['Concrete_mid']
R[21] = 1 / 2 * R_cd['Concrete_mid']
R[22] = Rw1.iloc[0]['out'] + Rw2.iloc[0]['out'] + Rroof.iloc[0]['out']
R[23] = 1 / 2 * (R_cd['Insulation1'] + R_cd['Insulation2'] +
                 R_cd['Insulation_roof'])
R[24] = 1 / 2 * (R_cd['Insulation1'] + R_cd['Insulation2'] +
                 R_cd['Insulation_roof'] + R_cd['Concrete1'] +
                 R_cd['Concrete2'] + R_cd['Concrete_roof'])
R[25] = 1 / 2 * (R_cd['Concrete1'] + R_cd['Concrete2'] + R_cd['Concrete_roof'])
R[26] = Rw1.iloc[0]['in'] + Rw2.iloc[0]['in'] + Rroof.iloc[0]['in']
R[27] = RLWGW
R[28] = Rg.iloc[0]['in']
R[29] = Rgs
R[30] = 1 / 2 * R_cd['Glass']
R[31] = Rv
R[32] = 1
R[33] = Rmid2
R[34] = RLWWF
R[35] = RLWGF
R[36] = RLWWC
R[37] = RLWGC
R[38] = RLWFC
G = np.diag(np.reciprocal(R))
G[17, 17] = Kp
G[32, 32] = Kp

# Capacities and Capacity Matrix
Cp = wall['Density'] * wall['Specific heat'] * wall['Surface'] * wall['Width']

Cp['Air'] = air['Density'] * air['Specific heat'] * Va

C = np.zeros(nds)
C[1] = Cp['Insulation1'] + Cp['Insulation2']
C[2] = Cp['Concrete1'] + Cp['Concrete2']
C[5] = Cp['Air']
C[7] = Cp['ground']
C[8] = Cp['ground']
C[9] = Cp['ground']
C[10] = Cp['Concrete_ground']
C[13] = Cp['Concrete_mid']
C[15] = Cp['Insulation1'] + Cp['Insulation2'] + Cp['Insulation_roof']
C[16] = Cp['Concrete1'] + Cp['Concrete2'] + Cp['Concrete_roof']
C[19] = Cp['Air']
C = np.diag(C)

# Temperature Source Vector b
b = np.zeros(brn)       # size defined by branches
b[[0, 7, 9, 17, 18, 22, 29, 31, 32]] = 1   # branches with temperature sources

# Heat flow source Vector f
f = np.zeros(nds)         # size defined by nodes
f[[0, 3, 6, 14, 17, 20]] = 1       # nodes with heat-flow sources

# Output Vector y
y = np.zeros(nds)       # size defined by nodes
y[[5, 19]] = 1              # nodes (temperatures) of interest


# State Space representation for the DAE
[As, Bs, Cs, Ds] = dm4bem.tc2ss(A, G, b, C, f, y)
print('As = \n', As, '\n')
print('Bs = \n', Bs, '\n')
print('Cs = \n', Cs, '\n')
print('Ds = \n', Ds, '\n')

# Steady State formulation
b = np.zeros(brn)                  # temperature sources
b[[0, 7, 9, 18, 22, 29, 31]] = 10  # outdoor temperature and ground temperature
b[[17, 32]] = 20                   # indoor set-point temperature

f = np.zeros(nds)         # flow-rate sources

# System of Diferential Algebraic Equations (DAE)
θ = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)
print(f'θ = {θ} °C')
# State Space representation of u
bT = np.array([10, 10, 10, 20, 10, 10, 10, 10, 20])
# [To, To, To, Tisp1, To, To, To, To, Tisp2]
fQ = np.array([0, 0, 0, 0, 0, 0])         # [Φo, Φi, Φa, Φo, Φi, Φa]
u = np.hstack([bT, fQ])
print(f'u = {u}')
yss = (-Cs @ np.linalg.inv(As) @ Bs + Ds) @ u
print(f'yss = {yss} °C')
print(f'Max error between DAE and state-space: \
      {max(abs(θ[6] - yss)):.2e} °C')

# Implementing the Dynamic Simulation Part

λ = np.linalg.eig(As)[0]    # eigenvalues of matrix As
print('Time constants: \n', -1 / λ, 's \n')
print('2 x Time constants: \n', -2 / λ, 's \n')
dtmax = min(-2. / λ)
print(f'Maximum time step: {dtmax:.2f} s = {dtmax / 60:.2f} min')
dt = 4.5 * 60     # seconds
print(f'dt = {dt} s = {dt / 60:.0f} min')

# Settling Time
t_resp = 4 * max(-1 / λ)
print('Time constants: \n', -1 / λ, 's \n')
print(f'Settling time: {t_resp:.0f} s = {t_resp / 60:.1f} min \
= {t_resp / (3600):.2f} h = {t_resp / (3600 * 24):.2f} days')

# Building a Step response

duration = 3600 * 24 * 70            # seconds, larger than response time
n = int(np.floor(duration / dt))    # number of time steps
t = np.arange(0, n * dt, dt)        # time vector for n time steps

print(f'Duration = {duration} s')
print(f'Number of time steps = {n}')


# Input Vector definition for the Step response
u = np.zeros([len(bT) + len(fQ), n])
# u = # [To, To, TG, Tisp1, To, To, To, To, Tisp2, Φo, Φi, Φa, Φo, Φi, Φa]
u[[[0, 1, 4, 5, 6, 7]], :] = 10 * np.ones([6, n])    # To = 10 for n time steps
u[2, :] = 12 * np.ones([1, n])    # TG = 12 for n time steps
u[[3, 8], :] = 20 * np.ones([1, n])      # Tisp = 20 for n time steps

n_s = As.shape[0]                      # number of state variables
θ_exp = np.zeros([n_s, t.shape[0]])    # explicit Euler in time t
θ_imp = np.zeros([n_s, t.shape[0]])    # implicit Euler in time t

I = np.eye(n_s)                        # identity matrix

for k in range(n - 1):
    θ_exp[:, k + 1] = (I + dt * As) @\
        θ_exp[:, k] + dt * Bs @ u[:, k]
    θ_imp[:, k + 1] = np.linalg.inv(I - dt * As) @\
        (θ_imp[:, k] + dt * Bs @ u[:, k])

y_exp = Cs @ θ_exp + Ds @  u
y_imp = Cs @ θ_imp + Ds @  u

fig, ax = plt.subplots()
ax.plot(t / 3600, y_exp.T, t / 3600, y_imp.T)
ax.set(xlabel='Time [h]',
        ylabel='$T_i$ [°C]',
        title='Step input: To')
ax.legend(['Explicit1', 'Explicit2', 'Implicit1', 'Implicit2'])
plt.show()



# start_date = '2000-01-01 12:00:00'
# end_date = '2000-04-01 18:00:00'

# print(f'{start_date} \tstart date')
# print(f'{end_date} \tend date')

# filename = './weather_data/CHE_ZH_Zurich.Affoltern.066640_TMYx.2007-2021.epw'
# [data, meta] = dm4bem.read_epw(filename, coerce_year=None)
# weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
# del data

# weather.index = weather.index.map(lambda t: t.replace(year=2000))
# weather = weather[(
#     weather.index >= start_date) & (
#     weather.index < end_date)]

# surface_orientation = {'slope': 90,
#                         'azimuth': 0,
#                         'latitude': 47}

# surface_orientation2 = {'slope': 0,
#                         'azimuth': 0,
#                         'latitude': 47}
# albedo = 0.2
# rad_surf_W = dm4bem.sol_rad_tilt_surf(
#     weather, surface_orientation, albedo)  # Wall
# rad_surf_R = dm4bem.sol_rad_tilt_surf(
#     weather, surface_orientation2, albedo)  # Roof

# rad_surf_W['EtotWall'] = rad_surf_W.sum(axis=1)
# rad_surf_R['Etot'] = rad_surf_R.sum(axis=1)

# data = pd.concat([weather['temp_air'], rad_surf_W['EtotWall'],
#                   rad_surf_R['Etot']], axis=1)
# data = data.resample(str(dt) + 'S').interpolate(method='linear')
# data = data.rename(columns={'temp_air': 'To'})

# # Input conditions
# data['Ti1'] = 22 * np.ones(data.shape[0])
# data['Ti2'] = 20 * np.ones(data.shape[0])
# data['TG'] = 10 * np.ones(data.shape[0])

# To = data['To']
# Ti1 = data['Ti1']
# Ti2 = data['Ti2']
# TG = data['TG']
# Φo1 = α_wSW * (Sw1 + Sw2) * data['EtotWall']
# Φo2 = α_wSW * Sg * data['Etot'] + Φo1
# Φi = τ_gSW * α_wSW * wall['Surface']['Glass'] * data['EtotWall']
# Φa = α_gSW * wall['Surface']['Glass'] * data['EtotWall']

# u = pd.concat([To, To, TG, Ti1, To, To, To, To, Ti2, Φo1, Φi, Φa, Φo2, Φi, Φa],
#               axis=1)
# u.columns.values[[9, 10, 11, 12, 13, 14]] = ['Φo1', 'Φi',
#                                              'Φa', 'Φo2', 'Φi', 'Φa']

# # Initial Conditions
# θ_exp = 20 * np.ones([As.shape[0], u.shape[0]])

# for k in range(u.shape[0] - 1):
#     θ_exp[:, k + 1] = (I + dt * As) @ θ_exp[:, k]\
#         + dt * Bs @ u.iloc[k, :]

# y_exp = Cs @ θ_exp + Ds @ u.to_numpy().T
# q_HVAC1 = Kp * (data['Ti1'] - y_exp[0, :])
# q_HVAC2 = Kp * (data['Ti2'] - y_exp[0, :])

# t = dt * np.arange(data.shape[0])   # time vector

# fig, axs = plt.subplots(2, 1)
# # plot indoor and outdoor temperature
# axs[0].plot(t / 3600 / 24, y_exp[0, :], label='$T_{indoor1}$')
# axs[0].plot(t / 3600 / 24, y_exp[1, :], label='$T_{indoor2}$')
# axs[0].plot(t / 3600 / 24, data['To'], label='$T_{outdoor}$')
# axs[0].plot(t / 3600 / 24, data['TG'], label='$T_{Ground}$')
# axs[0].set(xlabel='Time [days]',
#            ylabel='Temperatures [°C]',
#            title='Simulation for weather')
# axs[0].legend(loc='upper right')

# # plot total solar radiation and HVAC heat flow
# axs[1].plot(t / 3600 / 24, q_HVAC1, label='$q_{HVAC1}$')
# axs[1].plot(t / 3600 / 24, q_HVAC2, label='$q_{HVAC2}$')
# axs[1].plot(t / 3600 / 24, data['Etot'], label='$Φ_{total}$')
# axs[1].set(xlabel='Time [days]',
#            ylabel='Heat flows [W]')
# axs[1].legend(loc='upper right')

# fig.tight_layout()
