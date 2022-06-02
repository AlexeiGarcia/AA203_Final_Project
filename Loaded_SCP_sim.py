# Python packages
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import matplotlib.animation as animation

# Custom functions for project
import Quadcopter_Functions as QF
from quad_animation import animate_2D_quad

# Instantiate planar quadrotor object
quad = QF.PlanarQuadrotor()

# TRAJECTORY GENERATION

# quadrotor trajectory optimization parameters
dt = 0.01                                   # Simulation time step - sec
t_final = 4.                                # Simulation run time - sec
n = 8                                       # state dimension
m = 2                                       # control dimension

# Control bounds
Tmin = quad.min_thrust_per_prop
Tmax = quad.max_thrust_per_prop

# SCP parameters
P = 200*np.eye(n)   # terminal state cost matrix
Q = np.eye(n)       # state cost matrix
R = (1/Tmax**2)*np.eye(m)  # control cost matrix
tol = 0.5          # convergence tolerance
max_iters = 100     # maximum number of SCP iterations

# Dynamics propagation
fd = QF.loaded_dynamics

# Solve the swing-up problem with SCP
t1 = np.arange(0., t_final + dt, dt)
N = t1.size - 1
s0 = np.array([0., 0., 0., 0., 0., 0., 0., 0.])       # initial hover state
s_goal1 = np.array([2., 3., 0., 0., 0., 0., 0., 0.])   # desired location w/ hover final velocity
u_goal1 = np.array([((quad.m_Q+quad.m_p)*quad.g/2),((quad.m_Q+quad.m_p)*quad.g/2)])    # desired hover control
s_traj1, u_traj1 = QF.generate_scp_trajectory(fd, P, Q, R, N, s_goal1, s0, tol, max_iters, dt, quad)

s_goal2 = np.array([4., 0., 0., 0., 0., 0., 0., 0.]) 
t2 = np.arange(t1[-1], t1[-1] + t_final + dt, dt)
s_traj2, u_traj2 = QF.generate_scp_trajectory(fd, P, Q, R, N, s_goal2, s_traj1[-1,:], tol, max_iters, dt, quad)

plt.plot(s_traj1[:,0],s_traj1[:,1])
plt.plot(s_traj2[:,0],s_traj2[:,1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Loaded Dynamics SCP Trajectory")
plt.show()

plt.plot(t1,s_traj1[:,3])
plt.plot(t2,s_traj2[:,3])
plt.xlabel("Time (s)")
plt.ylabel("Phi")
plt.title("Pendulum Angle")
plt.show()

plt.plot(t1[:-1],u_traj1[:,0])
plt.plot(t1[:-1],u_traj1[:,1])
plt.plot(t2[:-1],u_traj2[:,0])
plt.plot(t2[:-1],u_traj2[:,1])
plt.xlabel("Time (s)")
plt.ylabel("Propeller Thrust")
plt.legend(('T1','T2'))
plt.show()

t_tot = np.concatenate((t1,t2))
s_traj_tot = np.concatenate((s_traj1, s_traj2))

fig, anim = animate_2D_quad(t_tot, s_traj_tot, frameDelay=1)
#display.display(display.HTML(anim.to_html5_video()))
f = r"C:\Users\night\Documents\AA203_Final_Project/planar_quad_2.gif"
writergif = animation.PillowWriter(fps=30)
writervideo = animation.FFMpegWriter(fps=60)
# anim.save(f, writer=writergif)
plt.show()
