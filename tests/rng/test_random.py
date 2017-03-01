"""
Simple test of the random number generator used in PyLag, as suggested by
Ross and Sharples (2004). The test plots the growth in the variance of particle
positions through time, given a constant, uniform background diffusivity. All 
particles start out life at 0.0. Dispersion follows the relation:

x(n+1) = x(n) + sqrt(2*K*dt)*G(0,1) 

where x(n) is a particle's position at the nth time step, K is the diffusivity,
dt is the time step (in s) and G(0,1) is a random deviate drawn from a Gaussian
distribution with mean 0 and standard deviation 1.

Author:
-------
James Clark (PML)
"""

import pylag.random as random
import numpy as np
import matplotlib.pyplot as plt

# No. of particles to simulate
n_particles = 4000

# No. of time steps
n_time_steps = 1000

# Time step
dt = 1.0

# Diffusivity
K = 1.0/24.0

# Time
time = np.arange(0.0, dt*n_time_steps, dt)

# Seed the random number generator
random.seed()

# Array holding partice positions
displacement = np.empty((n_particles, n_time_steps))
for i in xrange(n_particles):
    x = 0.0
    for j in xrange(n_time_steps):
        displacement[i,j] = x

        x = x + np.sqrt(2.0*K*dt) * random.gauss(0.0, 1.0)

# Variance growth
var = np.var(displacement, axis=0)

# Theoretical predictions for the growth in variance
var_analytical = time/12.0
var_analytical_plus_sigma = var_analytical + np.sqrt(var_analytical)
var_analytical_minus_sigma = var_analytical - np.sqrt(var_analytical) 

plt.figure()
plt.plot(time, var, 'b')
plt.plot(time, var_analytical, 'r')
plt.plot(time, var_analytical_minus_sigma, '--r')
plt.plot(time, var_analytical_plus_sigma, '--r')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Variance')
plt.show()

