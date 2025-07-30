import numpy as np
import matplotlib.pyplot as plt
import scipy


class Qvntvs:
    def __init__(self, potential_barrier_electron=10000, potential_barrier_hole =10000, band_gap_well = 1.5,
                  m_e_barrier=1, m_e_well=1, m_h_barrier=1, m_h_well=1, biasing_voltage=3, well_width_nm=1.2,
                  barrier_width_nm=1.2, n_wells=1, n_intervals=2000):
        
        #Defining Constants Used in the Quantum Mechanics Calculations
        self.h_bar = 1.0545718e-34
        self.ev_to_J = 1.602176634e-19
        self.m_e = 9.10938356e-31
        self.e = 1.602176634e-19

        #Defining Potential Barriers
        self.V_barrier_electron = potential_barrier_electron #If you want to model infinite potential well, set this to a very high value
        self.V0_electron = self.V_barrier_electron * 1.602176634e-19

        self.V_barrier_hole = potential_barrier_hole #If you want to model infinite potential well, set this to a very high value
        self.V0_hole = self.V_barrier_hole * 1.602176634e-19


        #Defining the Spatial Coordinates of the Well and Barriers
        self.well_width_nm = well_width_nm
        self.barrier_width_nm = barrier_width_nm
        self.well_width = well_width_nm * 1e-9
        self.barrier_width = barrier_width_nm * 1e-9
        self.total_length = (n_wells + 1) * self.barrier_width + n_wells * self.well_width
        self.n_wells = n_wells

        #Discretizing the Space
        self.n_intervals = n_intervals
        self.x = np.linspace(0, self.total_length, n_intervals)
        self.dx = self.x[1] - self.x[0]

        self.well_start_index = np.argmin(np.abs(self.x - self.barrier_width))
        self.well_end_index = np.argmin(np.abs(self.x - (self.barrier_width + self.well_width)))

        #Defining the Electron Effective Masses
        self.m_e_well = m_e_well * self.m_e
        self.m_e_barrier = m_e_barrier * self.m_e

        #Defining the Hole Effective Masses
        self.m_h_well = m_h_well * self.m_e
        self.m_h_barrier = m_h_barrier * self.m_e

        #Defining the Biasing Voltage
        self.biasing_voltage = biasing_voltage
        self.force = self.biasing_voltage / self.barrier_width

        #Defining Band Gap of the Well Material
        self.band_gap_well = band_gap_well * self.ev_to_J  # Band gap in Joules

    def rectangular_potential_profile(self, electron=True, plot=True):
        #Rectangular Potential Profile, multiple Quantum Well (MQW) Structures are added .v2
        #Both hole and electron potential profiles can be modeled using this function while remembering that
        #the hole  potential profile is the upside down version of the electron potential profile

        if electron == True:
            V_general = np.ones(self.n_intervals) * self.V0_electron

        else:
            V_general = np.ones(self.n_intervals) * self.V0_hole

        position = self.barrier_width #The rightmost position of the first barrier that gives the start of the first well

        for _ in range(self.n_wells):
            #Setting the barrier region
 
            left_of_well = position
            right_of_well = position + self.well_width

            left_of_well_index = np.argmin(np.abs(self.x - left_of_well))
            right_of_well_index = np.argmin(np.abs(self.x - right_of_well))

            V_general[left_of_well_index:right_of_well_index] = 0  

            position = right_of_well + self.barrier_width # Giving the new start position for the well (its leftmost position)

        if plot:
            if electron:
                plt.plot(self.x * 1e9, V_general / self.ev_to_J, label='Multiple Quantum Well, Electron', color='red')
                plt.xlabel("Position (nm)")
                plt.ylabel("Potential (eV)")
                plt.title(f"{self.n_wells} Quantum Wells, Electrons")
                plt.grid(True)
                plt.legend()
                plt.show()
            else:
                plt.plot(self.x * 1e9, -V_general / self.ev_to_J, label='Multiple Quantum Well, Hole', color='red')
                plt.xlabel("Position (nm)")
                plt.ylabel("Potential (eV)")
                plt.title(f"{self.n_wells} Quantum Wells, Holes")
                plt.grid(True)
                plt.legend()
                plt.show()

        else:
            plt.close()

        return V_general


    def triangular_potential_profile(self, electron=True, barrier_bending=True, plot=True, serial_print=False):
    #Rectangular Potential Profile, multiple Quantum Well (MQW) Structures are added .v2
    #Both hole and electron potential profiles can be modeled using this function while remembering that
    #the hole  potential profile is the upside down version of the electron potential profile

        if electron:
            V_general = np.ones(self.n_intervals) * self.V0_electron

        else:
            V_general = np.ones(self.n_intervals) * self.V0_hole

        position = self.barrier_width  # Start of the first well corresponds to the rightmost position of the first barrier

        for _ in range(self.n_wells):
            #Iterating n_wells times to create multiple wells
            
            left_of_well = position
            right_of_well = position + self.well_width

            left_of_well_index = np.argmin(np.abs(self.x - left_of_well))
            right_of_well_index = np.argmin(np.abs(self.x - right_of_well))

            #Discretized coordinates of the well
            x_well = self.x[left_of_well_index:right_of_well_index]

            #Calculating the potential at the wells 
            if electron:
                V_general[left_of_well_index:right_of_well_index] = self.e * self.force * (x_well - x_well[0])
            else:
                V_general[left_of_well_index:right_of_well_index] = -self.e * self.force * (x_well - x_well[0])

            if barrier_bending:
                #Calculating potentials at the left and right barriers
    
                left_barrier_start = left_of_well - self.barrier_width
                left_barrier_end = left_of_well
                left_barrier_start_index = np.argmin(np.abs(self.x - left_barrier_start))
                left_barrier_end_index = left_of_well_index #It ends at the left of the well

                left_barrier_positions = self.x[left_barrier_start_index:left_barrier_end_index]

                if electron:
                    V_general[left_barrier_start_index:left_barrier_end_index] = self.V0_electron + self.e * self.force * (left_barrier_positions - x_well[0])
                else:
                    V_general[left_barrier_start_index:left_barrier_end_index] = self.V0_hole - self.e * self.force * (left_barrier_positions - x_well[0])
                
                #The same for the right barrier
                
                right_barrier_start = right_of_well #It naturally starts at the right of the well
                right_barrier_end = right_of_well + self.barrier_width
                right_barrier_start_index = right_of_well_index
                right_barrier_end_index = np.argmin(np.abs(self.x - right_barrier_end))

                right_barrier_positions = self.x[right_barrier_start_index:right_barrier_end_index]
                if electron:
                    V_general[right_barrier_start_index:right_barrier_end_index] = self.V0_electron + self.e * self.force * (right_barrier_positions - x_well[-1])
                else:
                    V_general[right_barrier_start_index:right_barrier_end_index] = self.V0_hole - self.e * self.force * (right_barrier_positions - x_well[-1])

            position = right_of_well + self.barrier_width #Iterate to the next well

        if plot:
            if electron:
                plt.plot(self.x * 1e9, V_general / self.ev_to_J, label='Multiple Quantum Well, Electron', color='red')
                plt.xlabel("Position (nm)")
                plt.ylabel("Potential (eV)")
                plt.title(f"{self.n_wells} Quantum Wells, Electrons")
                plt.grid(True)
                plt.legend()
                plt.show()
            else:
                plt.plot(self.x * 1e9, -V_general / self.ev_to_J, label='Multiple Quantum Well, Hole', color='red')
                plt.xlabel("Position (nm)")
                plt.ylabel("Potential (eV)")
                plt.title(f"{self.n_wells} Quantum Wells, Holes")
                plt.grid(True)
                plt.legend()
                plt.show()

        else:
            plt.close()

        return V_general



    def effective_mass_profile(self, electron=True, plot=True):
        #Effective Mass Profile
        #Both hole and electron effective mass profiles can be modeled using this function.
        if electron:
            m_general = np.ones(self.n_intervals) * self.m_e_barrier

        elif electron == False:
            m_general = np.ones(self.n_intervals) * self.m_h_barrier

        position = self.barrier_width  #Same logic with the potentials

        for _ in range(self.n_wells):
        #Structure
            left_of_well = position
            right_of_well = position + self.well_width

            left_of_well_index = np.argmin(np.abs(self.x - left_of_well))
            right_of_well_index = np.argmin(np.abs(self.x - right_of_well))

            if electron:
                m_general[left_of_well_index:right_of_well_index] = self.m_e_well
            else:
                m_general[left_of_well_index:right_of_well_index] = self.m_h_well

            position = right_of_well + self.barrier_width #Iterate to the next well

        if plot:
            plt.plot(self.x * 1e9, m_general / self.m_e, label='Effective Mass Profile (in Free Electron Mass)', color='blue')
            plt.title(f"Effective Mass Profile - {'Electron' if electron else 'Hole'}")
            plt.xlabel("Position (nm)")
            plt.ylabel("Effective Mass (In Free Electron Mass)")
            plt.grid(True)
            plt.legend()
            plt.show()
        else:
            plt.close()
            
        return m_general

    def inverse_mass_profile(self, m_general, electron = True, plot=True):
        #Both hole and electron inverse mass profiles can be modeled using this function.

        if m_general is None:
            #Using our previously defined effective mass profile function to get the effective mass profile
            if electron:
                m_general = self.effective_mass_profile(electron=True, plot=False)
            elif electron == False:
                m_general = self.effective_mass_profile(electron=False, plot=False)

        inv_m_general = 1 / m_general
        inv_mass = np.zeros(self.n_intervals - 1)
        inv_mass = 0.5 * (inv_m_general[:-1] + inv_m_general[1:])

        #Plotting the Inverse Mass Profile
        if plot:
            plt.plot(self.x[:-1] * 1e9, inv_mass * self.m_e, label='Inverse Mass Profile', color='green')
            plt.title("Inverse Mass Profile (In Terms of Free Electron Mass)")
            plt.xlabel("Position (nm)")
            plt.ylabel("Inverse Mass (1/m_e)")
            plt.grid(True)
            plt.legend()
            plt.show()
        else:
            plt.close()

        return inv_mass

    def hamiltonian_matrix(self, V_general, inv_mass, electron=True, plot=True):

        #Negative Definite Laplacian Matrix for the Hamiltonian
        #Using the finite difference method to approximate the second derivative
        main_diagonal = np.zeros(self.n_intervals)
        off_diagonals = np.zeros(self.n_intervals-1)#Initializing the matrices

        off_diagonals[:] = -self.h_bar**2 / (self.dx**2) * inv_mass / 2 #Replacing all values with the computed values for the off-diagonal elements 

        main_diagonal[1:-1] = self.h_bar**2 / (self.dx**2) * (inv_mass[1:] + inv_mass[:-1]) / 2 + V_general[1:-1] #For middle points 
        main_diagonal[0] = self.h_bar**2 / (self.dx**2) * inv_mass[0] / 2 + V_general[0] #For the leftmost spatial point
        main_diagonal[-1] = self.h_bar**2 / (self.dx**2) * inv_mass[-1] / 2 + V_general[-1] #For the rightmost spatial point

        H = np.diag(main_diagonal) + np.diag(off_diagonals, 1) + np.diag(off_diagonals, -1) #Hamiltonian Matrix Approximation

        return H

    def eigen_equation(self, H, electron = True, plot=True, n_levels=4):
        #The eigenequation of Time-Independent Schrödinger Equation gives us the energy levels as eigenvalues and the wavefunctions as eigenvectors
        #Plotting the energy levels and wavefunctions
        energy_levels, wave_functions = scipy.linalg.eigh(H)

        if electron == False:
            energy_levels = -energy_levels
            wave_functions = -wave_functions 

        if plot == True:
            for i in range(n_levels):
                print(f"  Energy Level {i+1}: {(energy_levels[i]) / self.ev_to_J:.3f} eV")
                plt.title("Energy Level Plot")
                plt.xlabel("Position (nm)")
                plt.ylabel("Energy (eV)")
                plt.plot(self.x * 1e9, energy_levels[i] / self.ev_to_J * np.ones_like(self.x), label=f"Energy Level {i+1} ({energy_levels[i] / self.ev_to_J:.3f} eV)", color='blue')
                plt.legend()
            plt.grid(True)
            plt.figure()
            for n in range(n_levels):
                psi = wave_functions[:, n]
                psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, self.x)) 
                plt.plot(self.x * 1e9, psi, label=f"Wavefunction {n+1} (Energy: {energy_levels[n] / self.ev_to_J:.3f} eV)")
                plt.title("Wavefunction Plot")
                plt.xlabel("Position (nm)")
                plt.ylabel("Wavefunction (psi)")
                plt.grid(True)
                plt.legend()
            plt.show()
        else:
            plt.close()

        return energy_levels, wave_functions

    def recombination_probability(self, wave_function_electron, wave_function_hole, plot=True):
        #Calculating the recombination probability density using the overlap integral of the wavefunctions
        
        psi_e = wave_function_electron[0]
        psi_h = wave_function_hole[0]

        #Normalizing the wavefunctions
        psi_e = psi_e / np.sqrt(np.trapezoid(np.abs(psi_e)**2, self.x))
        psi_h = psi_h / np.sqrt(np.trapezoid(np.abs(psi_h)**2, self.x))
        
        #Calculating the overlap integral
        overlap = np.trapezoid(np.conj(psi_e) * psi_h, self.x)

        recombination_probability = np.abs(overlap)**2
        recombination_density = np.abs(np.conj(psi_e) * psi_h)**2

        print(f"Recombination Probability (Ground States): {recombination_probability:.3e}")

        if plot:
            plt.bar([0], [recombination_probability], color='purple', width4)
            plt.title("Total Recombination Probability")
            plt.ylabel("Probability Value")
            plt.grid(True)
            plt.show()

            plt.plot(self.x * 1e9, recombination_density, color='purple', label='Overlap Density')
            plt.title("Spatial Wavefunctions Overlap Density")
            plt.xlabel("Position (nm)")
            plt.ylabel("Overlap Density")
            plt.grid(True)
            plt.legend()
            plt.show()

        else:
            plt.close()

        return recombination_probability, recombination_density
        



