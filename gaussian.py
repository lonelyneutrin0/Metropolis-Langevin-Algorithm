import numpy as np 
from mala import Sampler, Mode

mu = np.array([1.0, -2.0, 0.5])
sigma2 = np.array([0.5, 2.0, 1.5])
inv_sigma2 = 1.0 / sigma2

def log_prob(x): 
    dx = x - mu
    return -0.5 * float(np.sum(dx * dx * inv_sigma2))

def grad_log_prob(x):
    dx = x - mu
    return -dx * inv_sigma2

sampler = Sampler(
    mode=Mode.LOG_PROB,
    target_dist=log_prob,
    grad_fn=grad_log_prob,
    n_atoms=10,
    step_size=0.8
)

samples = sampler.sample(5_000, burn_in=200, thin=5)

X = samples[:, 0, :]

print("Acceptance", sampler.acceptance_rate())
print("Empirical mean:", np.mean(X, axis=0))
print("Target mean:", mu.ravel())
print("Empirical variance:", np.var(X, axis=0))
print("Target variance:", sigma2.ravel())

stats = sampler.get_autocorrelation_times(samples, dims=[0, 1, 2], method="ips", max_lag=200)
print("IAT per dim:", stats["tau"])
print("ESS per dim:", stats["ess"])

# Acceptance 0.5642063492063492
# Empirical mean: [ 0.99157693 -1.98804336  0.50263108]
# Target mean: [ 1.  -2.   0.5]
# Empirical variance: [0.47715953 1.91868902 1.49684581]
# Target variance: [0.5 2.  1.5]
# IAT per dim: [1.22293587 4.06108745 2.77072557]
# ESS per dim: [4088.52181437 1231.19732296 1804.58146043]