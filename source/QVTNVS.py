import numpy as np
import matplotlib.pyplot as plt
import scipy


class Qvntvs:
    """
    QVNTVS : Quantum Well Solver for Visualization and Simulation of Semiconductors

    -------------------------------------------------------------------------------------------
    Key Features:
        1-Multiple Options for Wells : Heterojunctions, Forward-Biased Triangular Wells, Multiple Wells, et cetera
        2-Energy Level and Wavefunction plots for both electrons and holes
        3-Recombination probabilities and transition energies/optical phenomena

    --------------------------------------------------------------------------------------------

    Developer : sairbarbaros (Barbaros Şair)
    """
    
    def __init__(self, potential_barrier_electron=10000, potential_barrier_hole =10000, band_gap_well = 1.5,
                  m_e_barrier=1, m_e_well=1, m_hh_barrier=1, m_hh_well=1, m_lh_barrier=0.3, m_lh_well=0.3, biasing_voltage=3, built_in_voltage = 2.5, well_width_nm=1.2,
                  barrier_width_nm=1.2, n_wells=1, n_intervals=2000):
        """
        Initialize the parameters for quantum mechanical calculations

        Parameters
        ---------------

        potential_barrier_electron : float
            Potential barrier height seen by electrons inside the well (in eV)
        
        potential_barrier_hole : float
            Potential barrier height seen by heavy holes inside the well (in eV)
        
        band_gap_well : float
            Bandgap energy of the well material (in eV)

        m_e_barrier : float
            Effective mass of electrons inside the barrier (in m0 [free electron mass])
        
        m_e_well : float
            Effective mass of electrons inside the well (in m0 [free electron mass])

        m_hh_barrier : float
            Effective mass of heavy holes inside the barrier (in m0 [free electron mass])

        m_hh_well : float
            Effective mass of heavy holes inside the barrier (in m0 [free electron mass])

        m_lh_barrier : float
            Effective mass of light holes inside the barrier (in m0 [free electron mass])

        m_lh_well : float
            Effective mass of light holes inside the barrier (in m0 [free electron mass])
            
        biasing_voltage : float
            Biasing voltage amplitude (in Volts)

        built_in_voltage : float
            Built-in voltage of the pn-junction (in Volts)

        well_width_nm : float
            Spatial width of the well (in nm)
        
        barrier_width_nm : float
            Spatial width of the barrier (in nm)

        n_wells : integer
            Number of quantum wells inside the structure

        n_intervals : integer
            Number of intervals that spatial axis will be divided  
     
        """
        #Defining Constants Used in the Quantum Mechanics Calculations
        self.h_bar = 1.0545718e-34
        self.ev_to_J = 1.602176634e-19
        self.m_e = 9.10938356e-31
        self.e = 1.602176634e-19
        self.c = 3.0e8

        #Defining Potential Barriers
        self.V_barrier_electron = potential_barrier_electron #To model an infinite potential well, set this to a very high value
        self.V0_electron = self.V_barrier_electron * 1.602176634e-19

        self.V_barrier_hole = potential_barrier_hole #To model an infinite potential well, set this to a very high value
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
        self.m_hh_well = m_hh_well * self.m_e
        self.m_hh_barrier = m_hh_barrier * self.m_e

        self.m_lh_well = m_lh_well * self.m_e
        self.m_lh_barrier = m_lh_barrier * self.m_e

        #Defining the Biasing Voltage
        self.biasing_voltage = biasing_voltage
        self.built_in_voltage = built_in_voltage
        self.E_field = (self.built_in_voltage-self.biasing_voltage) /self.total_length
        

        #Defining Band Gap of the Well Material
        self.band_gap_well = band_gap_well*self.ev_to_J  # Band gap in Joules

        self.band_barrier = self.band_gap_well + self.V0_electron + self.V0_hole

    def rectangular_potential_profile(self, electron=True, plot=True):
        
        """
        Set the rectangular potential structure independently for electrons and holes
        
        Parameters
        ----------
        electron : boolean
            Set the particle experiencing the potential
        
        plot : boolean
            Set the plotting option
        
        Returns
        ----------
        V_general : ndarray
            Potential profile of the structure

        """

        if electron == True:
            V_general = np.ones(self.n_intervals) * self.V0_electron

        else:
            V_general = np.ones(self.n_intervals) * self.V0_hole

        position = self.barrier_width #The rightmost position of the first barrier, the start of the first well

        for _ in range(self.n_wells):
            #Setting the barrier region
 
            left_of_well = position
            right_of_well = position + self.well_width

            left_of_well_index = np.argmin(np.abs(self.x - left_of_well))
            right_of_well_index = np.argmin(np.abs(self.x - right_of_well))

            V_general[left_of_well_index:right_of_well_index] = 0  

            position = right_of_well + self.barrier_width #The rightmost position of the next barrier, the start of the next well

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


    def triangular_potential_profile(self, electron=True, plot=True):

        """
        Set the triangular potential structure mimicking forward-biasing independently for electrons and holes
        
        Parameters
        ----------
        electron : boolean
            Set the particle experiencing the potential
        
        plot : boolean
            Set the plotting option
        
        Returns
        ----------
        V_general : ndarray
            Potential profile of the structure (in Joules)

        """

        if electron == True:
            V_general =- np.ones(self.n_intervals) * self.e*self.E_field*self.x
            V_comp = V_general

        else:
            V_general = np.ones(self.n_intervals) *self.e*self.E_field*self.x
            V_comp = V_general
        position = self.barrier_width #The rightmost position of the first barrier, the start of the first well

        well_regions = []
        for _ in range(self.n_wells):
                #Setting the barrier region
    
            left_of_well = position
            right_of_well = position + self.well_width

            left_of_well_index = np.argmin(np.abs(self.x - left_of_well))
            right_of_well_index = np.argmin(np.abs(self.x - right_of_well))
            well_regions.append((left_of_well_index, right_of_well_index
                                 ))
            if electron:
                V_general[left_of_well_index:right_of_well_index] = V_comp[left_of_well_index:right_of_well_index] - self.V0_electron 
            else:
                V_general[left_of_well_index:right_of_well_index] = V_comp[left_of_well_index:right_of_well_index] - self.V0_hole
            position = right_of_well + self.barrier_width #The rightmost position of the next barrier, the start of the next well

        if self.built_in_voltage >= self.biasing_voltage:

            if electron:
                V_general -= V_general[-1]
                V_general += self.V0_electron

            else:
                V_general -= V_general[0]
                V_general += self.V0_hole

            

        else:

            if electron:
                V_general -= V_general[-1]
                V_general += self.V0_electron
            else:
                V_general -= V_general[0]
                V_general += self.V0_hole
        
        """
        The reference point for the electron potential profile is the conduction band of the n-side
        The reference point for the hole potential profile is the valence band of the p-side.
        The potential energy difference between the references is: Bandgap Energy_Well + qV_biasing  - qV_built_in
        """

        if plot:
            if electron:
                plt.plot(self.x * 1e9, V_general / self.ev_to_J, label='Potential Profile', color='red')
                plt.xlabel("Position (nm)")
                plt.ylabel("Potential (eV)")
                plt.title(f"Potential Profile of Electrons w.r.t. Conduction Band of N-Side (Well) ")
                plt.grid(True)
                plt.legend()
                plt.show()
            else:
                plt.plot(self.x * 1e9, -V_general / self.ev_to_J, label='Multiple Quantum Well, Hole', color='red')
                plt.xlabel("Position (nm)")
                plt.ylabel("Potential (eV)")
                plt.title("Potential Profile of Holes w.r.t. Valence Band of P-Side (Well))")
                plt.grid(True)
                plt.legend()
                plt.show()

        else:
            plt.close()

        return V_general
    
    
    def effective_mass_profile(self, plot=True):
        """
        Set the effective mass profiles for electrons and holes in different materials and heterojunctions
        
        Parameters
        ----------
        electron : boolean
            Set the particle experiencing the potential

        heavy_hole : boolean
            Set the type of hole

        plot : boolean
            Set the plotting option
        
        Returns
        -----------
        m_general_e : ndarray
            Effective mass profile of electrons
        
        m_general_hh : ndarray
            Effective mass profile of heavy holes
        
        m_general_lh : ndarray
            Effective mass profile of light holes
            
        """

        m_general_e = np.ones(self.n_intervals) * self.m_e_barrier
        m_general_hh = np.ones(self.n_intervals) * self.m_hh_barrier
        m_general_lh = np.ones(self.n_intervals) * self.m_lh_barrier

        position = self.barrier_width  #Same Idea with the Potentials

        for _ in range(self.n_wells):
        #Structure
            left_of_well = position
            right_of_well = position + self.well_width

            left_of_well_index = np.argmin(np.abs(self.x - left_of_well))
            right_of_well_index = np.argmin(np.abs(self.x - right_of_well))

            
            m_general_e[left_of_well_index:right_of_well_index] = self.m_e_well
    
            m_general_hh[left_of_well_index:right_of_well_index] = self.m_hh_well

            m_general_lh[left_of_well_index:right_of_well_index] = self.m_lh_well

            position = right_of_well + self.barrier_width #Iterate to the next well

        if plot:
        
                plt.plot(self.x * 1e9, m_general_e / self.m_e, label='Effective Mass Profile (in Free Electron Mass)', color='blue')
                plt.title(f"Effective Mass Profile of Electrons")
                plt.xlabel("Position (nm)")
                plt.ylabel("Effective Mass of Electrons (In Free Electron Mass)")
                plt.grid(True)
                plt.legend()
                plt.show()
    
                plt.plot(self.x * 1e9, -m_general_hh / self.m_e, label='Effective Mass Profile (in Free Electron Mass)', color='blue')
                plt.title(f"Effective Mass Profile of Heavy Holes")
                plt.xlabel("Position (nm)")
                plt.ylabel("Effective Mass (In Free Electron Mass)")
                plt.grid(True)
                plt.legend()
                plt.show()
                plt.plot(self.x * 1e9, -m_general_lh / self.m_e, label='Effective Mass Profile (in Free Electron Mass)', color='blue')
                plt.title(f"Effective Mass Profile of Light Holes")
                plt.xlabel("Position (nm)")
                plt.ylabel("Effective Mass (In Free Electron Mass)")
                plt.grid(True)
                plt.legend()
                plt.show()
        else:
            plt.close()
            
        return m_general_e, m_general_hh, m_general_lh

    def inverse_mass_profile(self, m_general, plot=True):
        """
        Compute the inverse masses and harmonic means for interfaces
        
        Parameters
        ----------
        m_general : ndarray
            Effective mass profile
        
        electron : boolean
            Set True if the profile is of electron

        plot : boolean
            Set True to plot

        Results
        -------

        inv_mass : ndarray
            Inverse mass profile
        """

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

    def hamiltonian_matrix(self, V_general, inv_mass):
        """
        Construct the Hamiltonian matrix numerically regarding Ben-Daniel-Duke and Finite-Difference Methods 
        
        Parameters
        ----------

        V_general : ndarray
            Potential profile of the structure
        
        inv_mass : ndarray
            Inverse effective mass profile

        Results
        ----------

        H : ndarray
            Hamiltonian matrix 
        """
        
        main_diagonal = np.zeros(self.n_intervals)
        off_diagonals = np.zeros(self.n_intervals-1)#Initializing the matrices

        off_diagonals[:] = -self.h_bar**2 / (self.dx**2) * inv_mass / 2 #Spatial boundaries
        main_diagonal[1:-1] = self.h_bar**2 / (self.dx**2) * (inv_mass[1:] + inv_mass[:-1]) / 2 + V_general[1:-1] #For middle points 
        main_diagonal[0] = self.h_bar**2 / (self.dx**2) * inv_mass[0] / 2 + V_general[0] #For the leftmost spatial point
        main_diagonal[-1] = self.h_bar**2 / (self.dx**2) * inv_mass[-1] / 2 + V_general[-1] #For the rightmost spatial point

        H = np.diag(main_diagonal) + np.diag(off_diagonals, 1) + np.diag(off_diagonals, -1) #Hamiltonian Matrix Approximation

        return H

    def eigen_equation(self, H, V_general, electron = True, heavy_holes = True, plot=True, n_levels=4):
        """
        Solve the eigenequation of Time-Independent Schrödinger Equation to get energy levels and wavefunctions

        Parameters
        --------------
        H : ndarray
            Hamiltonian Matrix
        
        V_general : ndarray
            Potential profile w.r.t. references

        electron : boolean
            Set True if the particle is electron

        heavy_holes : boolean
            Set True if the particle is a heavy hole

        plot : boolean
            Set True to plot

        n_levels : integer
            Maximum number of energy levels and wavefunctions to be computed.

        Results
        -------------
        bound_levels : ndarray
            Bound energy level states inside the well/wells

        bound_wavefunctions : ndarray
            Bound wavefunction states inside the well/wells

        """
        if electron:
            particle_name = "Electron"
        else:
            if heavy_holes:
                particle_name = "Heavy Holes"
            else:
                particle_name = "Light Hole"

        
        energy_levels, wave_functions = scipy.linalg.eigh(H)

        if (self.biasing_voltage == self.built_in_voltage):

            if electron == False:
                energy_levels = -energy_levels
                wave_functions = -wave_functions 
                V0 = self.V0_hole
            else:
                V0 = self.V0_electron
        else:

            if electron == False:
                energy_levels = -energy_levels
                wave_functions = wave_functions
                V0 = np.max(V_general)

            else:
                V0 = np.max(V_general) 
                
        bound_levels = []
        bound_wavefunctions = []
        
        for i in range(min(n_levels, len(energy_levels))):

            if abs(energy_levels[i]) < abs(V0):
                #if electron and energy_levels[i]>0:
                    bound_levels.append(energy_levels[i])
                    bound_wavefunctions.append(wave_functions[:, i])
                #elif not electron and energy_levels[i] < 0:
                   # bound_levels.append(energy_levels[i])
                    #bound_wavefunctions.append(wave_functions[:, i])

        bound_levels = np.array(bound_levels)
        bound_wavefunctions = np.column_stack(bound_wavefunctions) if bound_wavefunctions else np.array([])

        if len(bound_levels) == 0:
            print("No Bound Levels!")
            

        if plot == True:
            for i in range(len(bound_levels)):
                print(f" {particle_name} Energy Level {i+1}: {(bound_levels[i]) / self.ev_to_J:.3f} eV")
                plt.title(f"{particle_name} Energy Level Plot")
                plt.xlabel("Position (nm)")
                plt.ylabel("Energy (eV)")
                plt.plot(self.x * 1e9, bound_levels[i] / self.ev_to_J * np.ones_like(self.x), label=f" {particle_name} Energy Level {i+1} ({bound_levels[i] / self.ev_to_J:.3f} eV)", color='blue')
                plt.legend()
            plt.grid(True)
            plt.figure()
            for n in range(len(bound_levels)):
                psi = bound_wavefunctions[:, n]
                psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, self.x)) 
                plt.plot(self.x * 1e9, psi, label=f" {particle_name} Wavefunction {n+1} (Energy: {bound_levels[n] / self.ev_to_J:.3f} eV)")
                plt.title(f"{particle_name} Wavefunction Plot")
                plt.xlabel("Position (nm)")
                plt.ylabel("Wavefunction Amplitude (psi)")
                plt.grid(True)
                plt.legend()
            plt.show()
        else:
            plt.close()

        return bound_levels, bound_wavefunctions

    def recombination_probability(self, wave_function_electron, wave_function_hole, heavy_hole = True, plot=True):
        """
        Compute the recombination probabilities
        Note: Only Bound-State Recombinations can be computed

        Parameters
        ----------

        wave_function_electron : ndarray
            Spatial electron wavefunction inside the well/wells
        
        wave_function_hole : ndarray
            Spatial hole wavefunction inside the well/wells

        heavy_hole : boolean
            Set True if the hole is heavy.

        plot : boolean
            Set True to plot

        Returns
        -------

        recombination matrix : ndarray
            Recombination probabilities for each wavefunction combination

        """
        particle_1 = "Electron"
        if heavy_hole:
            particle_2 = "Heavy Hole"
        else:
            particle_2 = "Light Hole"
        

        if len(wave_function_electron) > 0 and len(wave_function_hole > 0):
            recombination_matrix = np.zeros((wave_function_electron.shape[1], wave_function_hole.shape[1]))
            for i in range(wave_function_electron.shape[1]):
                for j in range(wave_function_hole.shape[1]):
        
                    psi_e = wave_function_electron[:, i]  
                    psi_h = wave_function_hole[:, j]     
    
                    psi_e = psi_e / np.sqrt(np.trapezoid(np.abs(psi_e)**2, self.x))
                    psi_h = psi_h / np.sqrt(np.trapezoid(np.abs(psi_h)**2, self.x))
            
            #Calculating the overlap integral
                    overlap_magnitude = np.trapezoid(np.conj(psi_e) * psi_h, self.x)
                    recombination_prob = overlap_magnitude**2
                    recombination_matrix[i, j] = recombination_prob

            if plot:
                plt.imshow(recombination_matrix, cmap='viridis', interpolation='nearest')
                plt.colorbar(label='Value')  # shows value scale
                plt.title("Recombination Probabilities Heatmap")
                plt.xlabel(f"{particle_2}")
                plt.ylabel(f"{particle_1}")
                plt.show()

            else:
                plt.close()
        else:
            recombination_matrix = 0

            print("No Bound-State Recombination!")

        return recombination_matrix
        
    def transition_energies(self, energy_levels_electron, energy_levels_hole, recombination_matrix, heavy_hole = True, plot=True):
        """
        Calculate the transition energies
        Parameters
        ----------
        energy_levels_electron : ndarray
            Eigenvalues of the eigenequation with Hamiltonian for electrons (in Joules)
        
        energy_levels_hole : ndarray
            Eigenvalues of the eigenequation with Hamiltonian for holes (in Joules)

        recombination_matrix : ndarray
            Recombination probabilities for each wavefunction combination

        heavy_holes : boolean
            Set True if holes are heavy

        Returns
        --------

        ordered_optical_matrix : ndarray
            A matrix including transition energies and transition probabilities
        """
        particle_1 = "Electron"
        if heavy_hole:
            particle_2 = "Heavy Hole"
        else:
            particle_2 = "Light Hole"
        
        E_transition_matrix = np.zeros((len(energy_levels_electron), len(energy_levels_hole)))

        if len(energy_levels_electron) > 0 and len(energy_levels_hole) > 0:

            for i in range(len(energy_levels_electron)):
                for j in range(len(energy_levels_hole)):

                    E_photon = self.band_gap_well +self.biasing_voltage*self.ev_to_J - self.built_in_voltage*self.ev_to_J + (energy_levels_electron[i]) - energy_levels_hole[j]
                    if E_photon <0:
                        E_photon = 0
                        print("Warning! Unphysical Transition")

                    E_transition_matrix[i,j] = E_photon
    
        else:
            print("No confined states for either electron or holes")

        if plot:
                plt.imshow(E_transition_matrix, cmap='viridis', interpolation='nearest')
                plt.colorbar(label='Value')  # shows value scale
                plt.title("Transition Energies Heatmap")
                plt.xlabel(f"{particle_2}")
                plt.ylabel(f"{particle_1}")
                plt.show()

        else:
                plt.close()

        optical_matrix = []

        for Energy_Row, Prob_Row in zip(E_transition_matrix, recombination_matrix):
            row = []
            for Energy, Prob in zip(Energy_Row, Prob_Row):
                row.append((Energy, Prob))
            optical_matrix.append(row)

        flattened = [item for sublist in optical_matrix for item in sublist]

        ordered_optical_matrix = sorted(flattened, key=lambda x: x[1], reverse=True)

        return ordered_optical_matrix

    
    def optical_char(self, ordered_optical_matrix, heavy_hole = True):
        """
        Compute the emission wavelengths w.r.t. their probabilities

        Parameters
        ----------

        ordered_optical_matrix : ndarray
            Matrix containing recombination probability and transition energy information

        heavy_hole : boolean
            Set True if holes are heavy

        Returns
        ----------
        wavelengths : ndarray
            Contains emission wavelengths and the probabilities
        """
        
        wavelengths = []  
        particle_1 = "Electron"
        if heavy_hole:
            particle_2 = "Heavy Hole"
        else:
            particle_2 = "Light Hole"

        
        for energy, prob in ordered_optical_matrix:

            if prob > 0.3:
                if energy > 0:  # Avoid division by zero/unphysical energies
                    h = self.h_bar * 2 * np.pi  # Correct Planck's constant (h, not h-bar)
                    wavelength_m = h * self.c / energy  # Wavelength in meters
                    wavelength_nm = wavelength_m * 1e9  # Convert to nanometers
                    wavelengths.append((wavelength_nm, prob))
                else:
                    print(f"Warning! Unphysical Transition")

        if wavelengths:

            plt.figure()
            plt.title(f"{particle_1}-{particle_2} Emission Spectrum")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Recombination Probability")
            
            wavelength_vals = [w[0] for w in wavelengths]
            probs = [w[1] for w in wavelengths]
            
            plt.scatter(wavelength_vals, probs, color='blue',label='Optical Transitions')
            plt.grid(True)
            plt.legend()
            plt.show()
            
            # Print results
            for i, (wl, prob) in enumerate(wavelengths):
                print(f"Transition {i+1}: Wavelength = {wl:.2f} nm, Probability = {prob:.2%}")
        else:
            print("No valid transitions to plot.")

        return wavelengths