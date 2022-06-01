# Python packages
import numpy as np
import matplotlib as plt

# Custom functions for project
import Quadcopter_Functions as QF


# Instantiate planar quadrotor object
quad = QF.PlanarQuadrotor()

# TRAJECTORY GENERATION

# quadrotor trajectory optimization parameters
dt = 0.01                                   # Simulation time step - sec
t_final = 4.                                # Simulation run time - sec
n = 8                                       # state dimension
m = 2                                       # control dimension
s0 = np.array([0., 0., 0., 0., 0., 0., 0., 0.])       # initial hover state
s_goal = np.array([1., 1., 0., 0., 0., 0., 0., 0.])   # desired location w/ hover final velocity

# Control bounds
Tmin = quad.min_thrust_per_prop
Tmax = quad.max_thrust_per_prop

# SCP parameters
P = 150*np.eye(n)   # terminal state cost matrix
Q = np.eye(n)       # state cost matrix
R = (1/Tmax**2)*np.eye(m)  # control cost matrix
tol = 0.5          # convergence tolerance
max_iters = 100     # maximum number of SCP iterations

# Dynamics propagation
fd = QF.loaded_dynamics

# Solve the swing-up problem with SCP
t = np.arange(0., t_final + dt, dt)
N = t.size - 1
s_traj, u_traj = QF.generate_scp_trajectory(fd, P, Q, R, N, s_goal, s0, tol, max_iters, dt, quad)

plt.plot(s_traj[:,0],s_traj[:,1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Loaded Dynamics SCP Trajectory")
plt.show()

plt.plot(t[:-1],u_traj[:,0])
plt.plot(t[:-1],u_traj[:,1])
plt.xlabel("Time (s)")
plt.ylabel("Propeller Thrust")
plt.legend(('T1','T2'))
plt.show()
