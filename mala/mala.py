import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
import warnings

class MALA(MolecularDynamics):
    """
    Metropolis-Adjusted Langevin Algorithm (MALA) sampler.
    
    This implements MALA as a subclass of ASE's MolecularDynamics,
    allowing it to be used with the standard ASE interface while
    performing Bayesian sampling instead of deterministic dynamics.
    """
    
    def __init__(self, atoms, timestep, temperature=300*units.kB, 
                 step_size=None, trajectory=None, logfile=None, 
                 loginterval=1, append_trajectory=False):
        """
        Initialize MALA sampler.
        
        Parameters:
        -----------
        atoms : Atoms
            The atoms object to sample
        timestep : float
            Time step (here used as iteration counter for compatibility)
        temperature : float
            kT in energy units (default: 300K in eV)
        step_size : float, optional
            Proposal noise scale
        trajectory : str or Trajectory, optional
            Trajectory file to save samples
        logfile : str or file, optional
            Log file for energies and acceptance rates
        loginterval : int
            Interval for logging
        append_trajectory : bool
            Whether to append to existing trajectory
        """
        
        # Initialize parent MolecularDynamics class
        super().__init__(atoms, timestep, trajectory, logfile, loginterval, 
                        append_trajectory=append_trajectory)
        
        self.loginterval = loginterval
        
        self.temperature = temperature
        self.beta = 1.0 / temperature
        
        if step_size is None:
            self.step_size = np.sqrt(2.0 * temperature * timestep)
        else:
            self.step_size = step_size
            
        self.n_accepted = 0
        self.n_proposed = 0
        self.current_energy = None
        self.current_forces = None
        
        self._update_energy_forces()
    
    def _update_energy_forces(self):
        """Update current energy and forces."""
        self.current_energy = self.atoms.get_potential_energy()
        self.current_forces = self.atoms.get_forces()
    
    def _propose_move(self):
        """Generate MALA proposal."""
        positions = self.atoms.get_positions()
        forces = self.current_forces
        
        # MALA proposal: x' = x + ε²/2 * ∇log p(x) + ε * η
        # where ∇log p(x) = β * forces (since E = -log p up to const)
        drift = 0.5 * self.step_size**2 * self.beta * forces
        noise = self.step_size * np.random.normal(size=positions.shape)
        
        proposed_positions = positions + drift + noise
        
        return proposed_positions
    
    def _log_transition_probability(self, x_old, x_new, forces_old, forces_new):
        """
        Compute log transition probability ratio for MALA.
        
        log[q(x_old|x_new) / q(x_new|x_old)]
        """
        drift_forward = 0.5 * self.step_size**2 * self.beta * forces_old
        drift_backward = 0.5 * self.step_size**2 * self.beta * forces_new
        
        mean_reverse = x_new + drift_backward
        
        mean_forward = x_old + drift_forward
        
        # Detailed balance term
        diff_reverse = x_old - mean_reverse
        diff_forward = x_new - mean_forward
        
        log_q_ratio = -0.5 / self.step_size**2 * (
            np.sum(diff_reverse**2) - np.sum(diff_forward**2)
        )
        
        return log_q_ratio
    
    def _metropolis_step(self):
        """Perform one MALA step with Metropolis acceptance."""
        old_positions = self.atoms.get_positions().copy()
        old_energy = self.current_energy
        old_forces = self.current_forces.copy()
        
        proposed_positions = self._propose_move()
        
        self.atoms.set_positions(proposed_positions)
        try:
            proposed_energy = self.atoms.get_potential_energy()
            proposed_forces = self.atoms.get_forces()
        except Exception as e:

            self.atoms.set_positions(old_positions)
            warnings.warn(f"Energy calculation failed: {e}")
            return False
        
        log_alpha = (
            self.beta * (old_energy - proposed_energy) +
            self._log_transition_probability(
                old_positions, proposed_positions, old_forces, proposed_forces
            )
        )

        if log_alpha > 0 or np.random.rand() < np.exp(log_alpha):
            self.current_energy = proposed_energy
            self.current_forces = proposed_forces
            self.n_accepted += 1
            accepted = True
        else:
            self.atoms.set_positions(old_positions)
            accepted = False
        
        self.n_proposed += 1
        return accepted
    
    def step(self):
        """Perform one MALA sampling step."""
        accepted = self._metropolis_step()
        
        self.nsteps += 1
        
        return accepted
    
    def get_acceptance_rate(self):
        """Get current acceptance rate."""
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed
    
    def run(self, steps):
        """Run MALA sampling for specified number of steps."""
        for i in range(steps):
            accepted = self.step()
            
            if self.nsteps % self.loginterval == 0:
                acc_rate = self.get_acceptance_rate()
                if self.logfile is not None:
                    self.logfile.write(
                        f"{self.nsteps:6d} {self.current_energy:12.4f} "
                        f"{acc_rate:8.3f} {self.n_accepted:6d} {self.n_proposed:6d}\n"
                    )
                    self.logfile.flush()
                    
                if self.nsteps % 1000 == 0:
                    print(f"Step {self.nsteps}: Energy={self.current_energy:.4f}, "
                          f"AccRate={acc_rate:.3f}, Accepted={self.n_accepted}/{self.n_proposed}")
            
            if self.trajectory is not None:
                self.trajectory.write(self.atoms)
        
        final_acc_rate = self.get_acceptance_rate()
        return final_acc_rate