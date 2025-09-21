"""Metropolis Adjusted Langevin Algorithm (MALA) for MCMC sampling using ASE."""
from ase.md.langevin import Langevin
from ase import Atoms
from ase import units

from typing import Optional, List

import numpy as np
from numpy.typing import NDArray

class MALA(Langevin):
    accepts: List[bool]
    """The series of accepts."""
    rejects: List[bool]
    """The series of rejects."""

    @property
    def acceptance_ratio(self) -> float:
        """The acceptance ratio."""
        total = len(self.accepts) + len(self.rejects)

        if total == 0:
            return 0.0
        
        return sum(self.accepts) / total
    
    def __init__( 
            self,
            atoms: Atoms, 
            timestep: float, 
            friction: float,
            fixcm: bool = True, 
            *,
            temperature_K: Optional[float] = None,
            rng=None, 
            **kwargs,
    ):
        """
        Parameters
        ----------
        atoms: Atoms object
            The atoms to perform MALA on.

        timestep: float
            The time step  in ASE time units.

        temperature_K: float
            The temperature in Kelvin.

        friction: float
            A friction coefficient in inverse ASE time units.
        
        fixcm: bool, optional
            If True, the position and momentum of the 
            center of mass is kept unchanged. Default is True. 
        
        rng: RNG Object, optional
            A random number generator object. If None, the default 
            numpy random number generator is used.
        
        **kwargs:
            Additional arguments passed to the Langevin parent class.
"""
        
        super().__init__(
            atoms=atoms, 
            timestep=timestep, 
            temperature_K=temperature_K, 
            friction=friction, 
            fixcm=fixcm, 
            rng=rng, 
            **kwargs
        )
        self.temperature_K = temperature_K \
            if temperature_K is not None else 273.15

        self.accepts = [] 
        self.rejects = []
    
    def calculate_acceptance_probability(
        self,
        x_old: NDArray, 
        x_new: NDArray,
        forces_old: NDArray,
        forces_new: NDArray,
        energy_old: float,
        energy_new: float,
    ) -> float:
        r"""
        Calculate the acceptance probability for the MALA move.
        
        For overdamped Langevin dynamics, 
        $$\dd x = -\nabla U(x) \dd t + \sqrt{2/\beta} \dd W$$

        The acceptance probability naturally accounts for
        asymmetric proposals.
        """

        kT = units.kB * self.temperature_K

        drift_coefficient = self.dt / (self.masses[:, np.newaxis] * self.fr)
        diffusion_coefficient = 2 * self.dt * kT

        # Forward proposal: x_old -> x_new
        expected_move_forward = drift_coefficient * forces_old 
        actual_move = x_new - x_old
        noise_forward = actual_move - expected_move_forward

        # Reverse proposal: x_new -> x_old
        expected_move_backward = drift_coefficient * forces_new
        backward_move = x_old - x_new
        noise_backward = backward_move - expected_move_backward

        # Log of proposal probabilities
        variance = diffusion_coefficient / self.masses[:, np.newaxis]

        log_q_forward = -0.5 * np.sum((noise_forward**2) / variance)
        log_q_backward = -0.5 * np.sum((noise_backward**2) / variance)

        log_alpha = (-(energy_new - energy_old)/kT + 
                     log_q_backward - log_q_forward)

        return min(1.0, np.exp(log_alpha))

    def step(self, forces=None):
        """Perform one MALA step."""

        x_old = self.atoms.get_positions().copy()
        v_old = self.atoms.get_velocities().copy() \
            if self.atoms.get_velocities() is not None else None
        
        energy_old = self.atoms.get_potential_energy()

        if forces is None:
            forces_old =  self.atoms.get_forces(md=True)
        else:
            forces_old = forces.copy()
        
        forces_new = super().step(forces=forces_old)

        x_new = self.atoms.get_positions()
        energy_new = self.atoms.get_potential_energy()


        acceptance_prob = self.calculate_acceptance_probability(
            x_old=x_old,
            x_new=x_new,
            forces_old=forces_old,
            forces_new=forces_new,
            energy_old=energy_old,
            energy_new=energy_new,
        )

        if self.rng.random() < acceptance_prob:
            self.accepts.append(True)
            self.rejects.append(False)
            return forces_new

        else:
            self.atoms.set_positions(x_old)
            self.rejects.append(True)
            self.accepts.append(False)
            if v_old is not None:
                self.atoms.set_velocities(v_old)
        
            return forces_old
        
    def reset_statistics(self): 
        """Reset the acceptance/rejection statistics."""
        self.accepts = []
        self.rejects = []
        