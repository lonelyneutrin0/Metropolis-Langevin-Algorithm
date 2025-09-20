# Metropolis-adjusted Langevin Algorithm 
Metropolis-adjusted Langevin Algorithm ([MALA](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm)] is a Markov Chain Monte Carlo (MCMC) method to sample from intractable probability distributions. This algorithm proposes states using overdamped Langevin dynamics, and accepts/rejects them using the Metropolis-Hastings algorithm. The former directs the random walk towards high probability regions, while the latter is used to avoid localization and promote mixing.

I will be implementing this algorithm using [ASE](https://ase-lib.org/about.html)'s `Langevin` class.
