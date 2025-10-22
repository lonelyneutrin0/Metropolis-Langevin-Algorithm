"""MALA Sampler Backend"""

from typing import Optional, Callable, Union, Sequence
from numpy.typing import NDArray

import numpy as np 
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from mala.mala import MALA
from enum import Enum 

class Mode(Enum):
    ENERGY = "energy"
    LOG_PROB = "log_prob"

class CallableTargetCalculator(Calculator): 
    """ 
    Wraps user callables into an ASE Calculator interface. 
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        mode: Mode,
        kT: float,
        energy_fn: Optional[Callable[[NDArray], float]] = None,
        grad_energy_fn: Optional[Callable[[NDArray], NDArray]] = None,
        log_prob_fn: Optional[Callable[[NDArray], float]] = None,
        grad_log_prob_fn: Optional[Callable[[NDArray], NDArray]] = None,
        finite_diff: bool = False,
        fd_step: float = 1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.kT = float(kT)
        self.energy_fn = energy_fn
        self.grad_energy_fn = grad_energy_fn
        self.log_prob_fn = log_prob_fn
        self.grad_log_prob_fn = grad_log_prob_fn
        self.finite_diff = finite_diff
        self.fd_step = fd_step

        if mode == Mode.ENERGY: 
            if energy_fn is None: 
                raise ValueError("energy_fn must be provided in 'energy' mode.")
            if grad_energy_fn is None and not finite_diff: 
                raise ValueError("grad_energy_fn must be provided or finite_diff=True in 'energy' mode.")
        elif mode == Mode.LOG_PROB:
            if log_prob_fn is None: 
                raise ValueError("log_prob_fn must be provided in 'log_prob' mode.")
            if grad_log_prob_fn is None and not finite_diff: 
                raise ValueError("grad_log_prob_fn must be provided or finite_diff=True in 'log_prob' mode.")
        
    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = np.asarray(self.atoms.get_positions(), dtype=float)

        if self.mode == Mode.ENERGY:
            U = float(self.energy_fn(x))
            if self.grad_energy_fn is not None:
                 gradU = np.asarray(self.grad_energy_fn(x), dtype=float)
            else:
                gradU = self._fd_grad(x, self.energy_fn)
            if gradU.shape != x.shape:
                raise ValueError(f"grad_energy_fn must return shape {x.shape}, got {gradU.shape}")
            forces = -gradU
            self.results["energy"] = U
            self.results["forces"] = forces
            return 
        
        logp = float(self.log_prob_fn(x))
        if self.grad_log_prob_fn is not None:
            grad_logp = np.asarray(self.grad_log_prob_fn(x), dtype=float)
        else:
            def U_fn(xx): return -self.kT * float(self.log_prob_fn(xx))
            grad_logp = -1.0 / self.kT * self._fd_grad(x, U_fn)
        if grad_logp.shape != x.shape:
            raise ValueError(f"grad_log_prob_fn must return shape {x.shape}, got {grad_logp.shape}")

        U = -self.kT * logp
        forces = self.kT * grad_logp
        self.results["energy"] = U
        self.results["forces"] = forces

    def _fd_grad(self, x: NDArray, f: Callable[[NDArray], float]) -> NDArray:
        eps = self.fd_step
        grad = np.zeros_like(x, dtype=float)

        for i in range(x.shape[0]):
            for j in range(3): 
                x_plus = x.copy() 
                x_plus[i, j] += eps
                x_minus = x.copy()
                x_minus[i, j] -= eps
                grad[i, j] = (f(x_plus) - f(x_minus)) / (2 * eps)
        return grad
    
class Sampler: 
    """ 
    High-level sampler that accepts Callables 
    and runs MALA sampling on them.
    """

    def __init__(
        self,
        mode: Mode,
        target_dist: Callable[[NDArray], float],
        grad_fn: Optional[Callable[[NDArray], NDArray]] = None,
        *,
        kT: float = 300.0 * units.kB,
        n_atoms: int = 1,
        labels: Union[str, Sequence[str]] = "H",
        init_positions: Optional[NDArray] = None,
        step_size: float = 0.3,
        timestep: float = 1.0,
        finite_diff: bool = False,
        fd_step: float = 1e-6,
        mala_kwargs: Optional[dict] = None,
    ):
        
        if init_positions is None:
            init_positions = np.zeros((n_atoms, 3), dtype=float)
        else:
            init_positions = np.asarray(init_positions, dtype=float)
            if init_positions.shape != (n_atoms, 3):
                raise ValueError(f"init_positions must have shape {(n_atoms, 3)}, got {init_positions.shape}")
        if isinstance(labels, str): 
            symbols = [labels] * n_atoms
        else:
            symbols = list(labels)
            if len(symbols) != n_atoms:
                raise ValueError("Length of labels must match n_atoms.")
            
        atoms = Atoms(symbols, positions=init_positions)

        calc = CallableTargetCalculator(
            mode=mode,
            kT=kT,
            energy_fn=target_dist if mode == Mode.ENERGY else None,
            grad_energy_fn=grad_fn if mode == Mode.ENERGY else None,
            log_prob_fn=target_dist if mode == Mode.LOG_PROB else None,
            grad_log_prob_fn=grad_fn if mode == Mode.LOG_PROB else None,    
            finite_diff=finite_diff,
            fd_step=fd_step,
        )
        atoms.calc = calc
        self.atoms = atoms
        self.mala = MALA(
            atoms, 
            timestep=timestep,
            temperature=kT,
            step_size=step_size,
            **(mala_kwargs or {})
        )

    def step(self):
        return self.mala.step() 
    
    def sample(self, n: int, burn_in: int = 0, thin: int = 1) -> NDArray:
        samples = []
        total = burn_in + n * thin
        for i in range(total):
            self.step()
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(self.atoms.get_positions().copy())
        return np.asarray(samples)

    def acceptance_rate(self) -> float:
        return getattr(self.mala, "get_acceptance_rate", lambda: 0.0)()

    def get_autocorrelation_times(self, samples: NDArray,
        dims: Optional[Sequence[int]] = None, method: str = "ips",
        max_lag: Optional[int] = None) -> dict:
        """
        Compute per-dimension ACF, IAT, and ESS from samples

        Parameters: 
            samples : NDArray
                Array of shape (n_samples, n_atoms, 3)
            dims : Optional[Sequence[int]]
                Dimensions (0-2) to compute stats for. If None, all dimensions.
            method : str    
                Method for IAT estimation ("ips" or "acor")
            max_lag : Optional[int]
                Maximum lag to consider for ACF

        Returns:
            dict
                Dictionary with keys "acf", "tau", "ess"
        """
        X = np.asarray(samples) 

        if X.ndim != 3:
            raise ValueError("samples must be a 3D array of shape (n_samples, n_atoms, 3).")
        
        n = X.shape[0]

        if n < 2:
            raise ValueError("At least two samples are required to compute autocorrelation.")
        
        Y = X.reshape(n, -1)
        D = Y.shape[1]

        if dims is None: 
            dims = list(range(D))
        else:
            dims = list(dims)
            for d in dims:
                if d < 0 or d >= D:
                    raise ValueError(f"dim index {d} out of range [0, {D-1}]")
        
        taus, esses, acfs = [], [], [] 

        for d in dims:
            y = Y[:, d]
            acf = self._acf_fft(y, max_lag)
            tau = self._integrated_time(acf, method=method)
            ess = n / tau if tau > 0 else 0.0 
            taus.append(tau)
            esses.append(ess)
            acfs.append(acf)
        
        return {"acf": acfs, "tau": np.asarray(taus), "ess": np.asarray(esses), "dims": dims}
    
    @staticmethod
    def _acf_fft(x: NDArray, max_lag: Optional[int] = None) -> NDArray:
        """Compute normalized autocorrelation function using FFT.

        Returns acf with acf[0] == 1 and optional finite-sample bias correction.
        """
        x = np.asarray(x, dtype=float)
        x = x - np.mean(x) 
        n = x.size

        if n == 0:
            return np.array([1.0])
    
        nfft = 1 << (2 * n - 1).bit_length()
        f = np.fft.rfft(x, n=nfft)
        acf = np.fft.irfft(f * np.conjugate(f), nfft)[:n]
        # Finite-sample bias correction and normalization
        acf = acf / np.arange(n, 0, -1)
        c0 = acf[0]
        if c0 != 0:
            acf = acf / c0
        else:
            acf = np.ones_like(acf)

        if max_lag is not None:
            acf = acf[: max(1, int(max_lag) + 1)]
        
        return np.real(acf) 
    
    @staticmethod
    def _integrated_time(acf: NDArray, method: str = "ips") -> float:
        """Estimate integrated autocorrelation time from ACF."""
        if acf.size == 0: 
            return 1.0 
        
        if method == "ips":
            tau = 1.0

            k = 1
            while k < acf.size:
                pair = acf[k] + (acf[k + 1] if k + 1 < acf.size else 0.0)
                if pair <= 0.0:
                    break
                tau += 2.0 * pair
                k += 2
            return max(tau, 1.0)
        
        elif method == "positive":
            pos = acf[1:]
            m = np.argmax(pos <= 0.0)
            if (pos <= 0).any():
                cutoff = m
            else:
                cutoff = pos.size
            
            tau = 1.0 +  2.0 * float(np.sum(pos[:cutoff]))
            return max(tau, 1.0)
        else:
            raise ValueError(f"Unknown method '{method}' for IAT estimation.")