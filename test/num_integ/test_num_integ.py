import numpy as np
from matplotlib import pyplot as plt

from pylag.analytic_data_reader import AnalyticDataReader
from pylag.particle import Particle
from pylag.integrator import RK4Integrator

group_id = 0
x_0 = 0.1
y_0 = 0.1
z_0 = 0.0

test_particle = Particle(group_id, x_0, y_0, z_0)

data_reader = AnalyticDataReader()

time_start = 0.0
time_end = 3.0
time_step = 0.01

time = np.arange(time_start,time_end,time_step)

rk4 = RK4Integrator(time_step)

xpos_analytic = []
ypos_analytic = []
xpos_numeric = []
ypos_numeric = []

for t in time:
    xpos, ypos = data_reader.get_position_analytic(x_0, y_0, t)
    xpos_analytic.append(xpos)
    ypos_analytic.append(ypos)
    
    xpos_numeric.append(test_particle.xpos)
    ypos_numeric.append(test_particle.ypos)
    
    rk4.advect(t, test_particle, data_reader)
    
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
plt.title('RK4 test: particle position of a function of time.')
plt.legend()
plt.savefig('rk4_test.eps')