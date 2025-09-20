# Metropolis-adjusted Langevin Algorithm 
Metropolis-adjusted Langevin Algorithm ([MALA](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm)] is a Markov Chain Monte Carlo (MCMC) method to sample from intractable probability distributions. This algorithm proposes states using overdamped Langevin dynamics, and accepts/rejects them using the Metropolis-Hastings algorithm. The former directs the random walk towards high probability regions, while the latter is used to avoid localization and promote mixing.

I will be implementing this algorithm using [ASE](https://ase-lib.org/about.html)'s `Langevin` class.

## Installation 
### Direct installation from source
To install from the source directory, run 
```
pip install .
```

To install optional dependencies, such as `examples`, run the command

```
pip install .[examples]
```
### Build from distribution
To build this package locally, run the following command in the main project directory
```
pip -m pip install build
python -m build 
```

Then, to install the package, 
```
pip install dist/*.whl
```