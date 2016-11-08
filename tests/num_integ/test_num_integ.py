import numpy as np
from matplotlib import pyplot as plt
from ConfigParser import SafeConfigParser

from pylag.analytic_data_reader import TestVelocityDataReader
from pylag import cwrappers

# Create data reader
data_reader = TestVelocityDataReader()

# Timings
time_start = 0.0
time_end = 3.0
time_step = 0.01
time = np.arange(time_start,time_end,time_step)


# Config - needed for creation of the numerical integrator
config = SafeConfigParser()
config.add_section("SIMULATION")
config.set("SIMULATION", "time_step", str(time_step))
config.set("SIMULATION", "num_integrator", "RK4_2D")

# Create test object
test_num_integrator = cwrappers.TestRK4Integrator(config)

# Arrays in which particle position data will be stored
xpos_analytic = []
ypos_analytic = []
xpos_numeric = []
ypos_numeric = []

# Initial position
x_0 = 0.1
y_0 = 0.1
z_0 = 0.0

# Initialse variables that will hold the position of the particle as calculated
# by the chosen numerical integration scheme
x_numeric = x_0
y_numeric = y_0
z_numeric = z_0

for t in time:
    # Compute and save position using analytic formulae
    x_analytic, y_analytic = data_reader.get_position_analytic(x_0, y_0, t)
    xpos_analytic.append(x_analytic)
    ypos_analytic.append(y_analytic)
    
    # Save position as computed by the numerical scheme
    xpos_numeric.append(x_numeric)
    ypos_numeric.append(y_numeric)
    
    # Advect the particle
    x_numeric, y_numeric, z_numeric = test_num_integrator.advect(data_reader, t, x_numeric, y_numeric, z_numeric)
    
xmin = np.min(xpos_analytic)
xmax = np.max(xpos_analytic)
ymin = np.min(ypos_analytic)
ymax = np.max(ypos_analytic)

x_arr = np.linspace(xmin,xmax,10)
y_arr = np.linspace(ymin,ymax,10)

u_arr = []
for x in x_arr:
    u,v,w = data_reader.get_velocity_analytic(x, 0.0, 0.0)
    u_arr.append(u)

v_arr = []
for y in y_arr:
    u,v,w = data_reader.get_velocity_analytic(0.0, y, 0.0)
    v_arr.append(v)

u_grid, v_grid = np.meshgrid(u_arr, v_arr)

# Plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
qp = ax.quiver(x_arr, y_arr, u_grid, v_grid)
qk = ax.quiverkey(qp, 0.5, 0.92, 10, r'$10 m/s$', fontproperties={'weight': 'bold'})
ax.plot(xpos_analytic, ypos_analytic, linestyle='-', c='b', label='Analytic')
ax.plot(xpos_numeric[::5], ypos_numeric[::5], linestyle='None', marker='o', c='r', label='Numeric')
ax.grid()
ax.axhline(0, color='black', lw=2)
ax.axvline(0, color='black', lw=1)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.title('RK4 test: particle position as a function of time.')
plt.legend()
plt.savefig('rk4_test.eps')
