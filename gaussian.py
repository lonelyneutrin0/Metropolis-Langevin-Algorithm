import numpy as np 
from mala import Sampler, Mode
import matplotlib.pyplot as plt

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

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i in range(3):
    ax = axes[i]
    
    ax.hist(X[:, i], bins=40, density=True, alpha=0.6, color='steelblue', label='Sampled')
    
    x_range = np.linspace(X[:, i].min(), X[:, i].max(), 200)
    pdf = (1.0 / np.sqrt(2 * np.pi * sigma2[i])) * np.exp(-0.5 * ((x_range - mu[i])**2) / sigma2[i])
    ax.plot(x_range, pdf, 'r-', lw=2, label='Theoretical')
    
    ax.set_xlabel(f'Dimension {i}')
    ax.set_ylabel('Density')
    ax.set_title(f'Dim {i}: μ={mu[i]:.1f}, σ²={sigma2[i]:.1f}\nIAT={stats["tau"][i]:.2f}, ESS={stats["ess"][i]:.0f}')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('gaussian_sampling.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to gaussian_sampling.png")