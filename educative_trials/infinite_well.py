##INFINITE POTENTIAL WELL SOLVER (TO CHECK IF THE CODE WORKS)

import numpy as np
import scipy
import matplotlib.pyplot as plt

h_bar = 1.0545718e-34  # J.s
ev_to_J = 1.602176634e-19  # J/eV
m_e = 9.10938356e-31  # kg, free electron mass

#Object-Oriented Approach for v2

V_barrier = 10000 #Potential Barrier Height of InP-InGaAs heterojunction in eV [Tunable]

V0 = V_barrier * ev_to_J # Potential Barrier Height in Joules [Use in Calculations]

eff_e_well = 1 #Effective Mass of Electrons in InGaAs in units of free electron mass [m_e] [Tunable]
eff_e_barrier = 1 #Effective Mass of Electrons in InP in units of free electron mass [m_e] [Tunable]

m_e_well = eff_e_well * m_e #Effective Mass of Electrons in InGaAs in kg [Use in Calculations]
m_e_barrier = eff_e_barrier * m_e #Effective Mass of Electrons in InP in kg [Use in Calculations]

well_width_nm = 5 #Width of the well in nanometers [Tunable]
well_width = well_width_nm * 1e-9 #Width of the well in meters [Use in Calculations]

barrier_width_nm = 10 #Width of the barrier in nanometers [Tunable]
barrier_width = barrier_width_nm * 1e-9 #Width of the barrier in meters [Use in Calculations]

total_length = well_width + 2 * barrier_width #Total length of the heterojunction in meters [Use in Calculations]

N = 2000 #Number of intervals for the finite difference method [Tunable]

x = np.linspace(0, total_length, N) #Linear Space for the finite difference method [Use in Calculations]
dx = x[1] - x[0] #Increment size for the finite difference method [Use in Calculations]
#Defining the spatial coordinates of the well and barriers
well_starting_point = barrier_width #Floating point where the well starts [Will be Modified]
well_ending_point = well_starting_point + well_width #[Will be Modified]

well_start_index = np.argmin(np.abs(x - well_starting_point)) 
well_end_index = np.argmin(np.abs(x - well_ending_point)) #They are indexed to the nearest point in the linear space for numerical calculations 


#Potential Profile
V_general = np.ones(N) * V0 #Setting all the points (N amount) to the potential barrier height 
V_general[well_start_index:well_end_index] = 0 #The potential inside the well is naturally zero 

#Plotting the Potential Profile
plt.plot(x* 1e9, V_general / ev_to_J, label='Potential Profile (eV)', color='red')  
plt.xlabel("Position (nm)")
plt.ylabel("Potential (eV)")
plt.title("Infinite Potential Well Profile Width = 5 nm")
plt.grid(True)
plt.legend()
plt.figure()

#Effective Mass Profile
m_general = np.ones(N) * m_e_barrier #Setting all the points (N amount) to the effective mass of the barrier
m_general[well_start_index:well_end_index] = m_e_well #The effective mass inside the well 

#Plotting the Effective Mass Profile (Plotted in terms of free electron mass to show it easier)
plt.plot(x* 1e9, m_general/m_e, label='Effective Mass Profile (In terms of Free Electron Mass)', color='blue')
plt.title("Effective Mass Profile")
plt.xlabel("Position (nm)")
plt.ylabel("Effective Mass (m_e)")
plt.grid(True)
plt.legend()
plt.figure()

inv_m_general = 1 / m_general #Inverse Mass Profile for the generalized kinetic energy operator
inv_mass = np.zeros(N-1) #There are N-1 between points for N points in the linear space
inv_mass = 0.5 * (inv_m_general[:-1] + inv_m_general[1:]) #Averaging before and after values for each between point

#Plotting the Inverse Mass Profile
plt.plot(x[:-1]* 1e9, inv_mass*m_e, label='Inverse Mass Profile', color='green')
plt.title("Inverse Mass Profile (In Terms of Free Electron Mass)")   
plt.xlabel("Position (nm)")
plt.ylabel("Inverse Mass (1/m_e)")
plt.grid(True)
plt.legend()
plt.show()
plt.figure()

#Negative Definite Laplacian Matrix for the Hamiltonian
#Using the finite difference method to approximate the second derivative

main_diagonal = np.zeros(N)
off_diagonals = np.zeros(N-1)#Initializing the matrices

off_diagonals[:] = -h_bar**2 / (dx**2) * inv_mass / 2 #Replacing all values with the computed values for the off-diagonal elements 

main_diagonal[1:-1] = h_bar**2 / (dx**2) * (inv_mass[1:] + inv_mass[:-1]) / 2 + V_general[1:-1] #For middle points 
main_diagonal[0] = h_bar**2 / (dx**2) * inv_mass[0] / 2 + V_general[0] #For the leftmost spatial point
main_diagonal[-1] = h_bar**2 / (dx**2) * inv_mass[-1] / 2 + V_general[-1] #For the rightmost spatial point

H = np.diag(main_diagonal) + np.diag(off_diagonals, 1) + np.diag(off_diagonals, -1) #Hamiltonian Matrix Approximation

energy_levels, wave_functions = scipy.linalg.eigh(H) #The eigenvalues are the energy levels, and the eigenvectors are the wavefunctions

for i in range(4):
    print(f"  Energy Level {i+1}: {(energy_levels[i]) / ev_to_J:.3f} eV")
    plt.hlines(energy_levels[i]/ev_to_J, x[0]*1e9, x[-1]*1e9, color='r')
    plt.title("Infinite Potential Well Energy Levels Width = 5 nm")

plt.figure()

for n in range(4):
    psi = wave_functions[:, n]
    psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x)) 
    plt.plot(x * 1e9, psi, label=f"Wavefunction {n+1} (Energy: {energy_levels[n] / ev_to_J:.3f} eV)")
plt.legend()
plt.title("Infinite Potential Well Wavefunctions Width = 5 nm")
plt.grid(True)
plt.figure()
plt.show()
