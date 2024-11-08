import autograd.numpy as np
import numpy 
from dataclasses import dataclass
import inspect 
from typing import Callable
from autograd import grad 

@dataclass 
class out: 
    Xs: np.array 
    probs: np.array
    accepts: np.array
    rejects: np.array
    """
    Container class for output
    :param Xs: Samples
    :param probs: Probability series 
    :param accepts: cumsum of acceptances 
    :param rejects: cumsum of rejects 
    """

@dataclass 
class lmc: 
    probability_dist: Callable[[numpy.array], float]
    epsilon: float 
    num_samples: int
    d: int 
    X_0: numpy.array=None
    """
    Container class for the algorithm 
    
    :param probability_dist: The sampling function which takes a list of inputs
    :param epsilon: The step size
    :param num_samples: The number of samples desired
    :param d: Dimensionality
    :param X_0: The initial configuration X_0 to start with 
    """
    
    def sample(self): 
        # If an appropriate initial vector is not provided, start with a zero vector of appropriate dimension 
        self.X_0 = self.X_0 if (self.X_0 is not None and self.X_0.size == self.d) else np.ones((self.d,))
        
        # Initializations 
        Xs = []
        probabilities = [] 
        accepts = []
        rejects = []
        
        # Create the Langevin Distribution and the Langevin Gradient
        langevin_dist: Callable[[numpy.array], float] = lambda x: np.log(self.probability_dist(x))
        langevin_grad = grad(langevin_dist)
        
        # Start the markov chain
        while len(Xs) < self.num_samples: 
            # Propose a new state 
            X_new = self.X_0 + self.epsilon*langevin_grad(self.X_0) + np.sqrt(2*self.epsilon)*numpy.random.multivariate_normal(np.zeros(self.d), np.eye(self.d), 1).squeeze()
            # print(langevin_grad(self.X_0))
            # Compute the acceptance probability of the proposed solution 
            P_new = self.probability_dist(X_new)
            P_old = self.probability_dist(self.X_0)
            reverse_P = np.exp(-1/(4*self.epsilon) * np.linalg.norm(X_new - self.X_0 - self.epsilon * langevin_grad(self.X_0))**2)
            forward_P = np.exp(-1/(4*self.epsilon) * np.linalg.norm(self.X_0 - X_new - self.epsilon * langevin_grad(X_new))**2)
            rho = numpy.minimum(1, (P_new*forward_P)/(P_old*reverse_P))
            # If the acceptance ratio is greater than u ~ N(0,1) then add the configuration and transition to it
            if(rho >= numpy.random.uniform(low=0, high=1)): 
                probabilities.append(P_new)
                Xs.append(X_new)
                self.X_0 = X_new
                accepts.append(1)
                rejects.append(0)
            else: 
                rejects.append(1)
                accepts.append(0)
        
        outputargs = { 
            'Xs': numpy.array(Xs),
            'probs': numpy.array(probabilities), 
            'accepts': numpy.array(accepts), 
            'rejects': numpy.array(rejects)
        }
        return out(**outputargs)
