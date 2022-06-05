import numpy as np
from numpy.random import rand

def mmc(p,x0,delta=1):
    """Samples the next point from p using Metropolis Monte Carlo.

    - p         a function that takes a vector and returns some value
                proportional to probability distribution to sample.

    - x0        the prior point. This must be the same size as what p takes.

    - delta     (optional) the length of the hypercube to sample from as in
                the metropolis update. (default: 1)
    """
    #metropolis update
    if hasattr(x0,"__len__"): #has len() attrib, assume it is a vector
        x1 = np.array(x0) + (rand(len(x0)) - 0.5)*delta
    else: #has no len(), assume it is a single value
        x1 = x0 + (rand() - 0.5)*delta
    a = rand() # acceptance parameter

    return x1 if a < p(x1)/p(x0) else x0

