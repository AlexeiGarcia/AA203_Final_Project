# Python packages
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import matplotlib.animation as animation

# Custom functions for project
import Quadcopter_Functions as QF
from Tracking_LQR import LQR_tracking_gain
from quad_animation import animate_2D_quad

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
u_goal = np.array([((quad.m_Q+quad.m_p)*quad.g/2),((quad.m_Q+quad.m_p)*quad.g/2)])    # desired hover control

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
t = np.arange(0., t_final + dt, dt)
N = t.size - 1
# s_traj, u_traj = QF.generate_scp_trajectory(fd, P, Q, R, N, s_goal, s0, tol, max_iters, dt, quad)
'''
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
'''

# --- Test LQR control with loaded dynamics ---

waypoints = np.array([[5., 5., 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0., 0.]])

num_wp = waypoints.shape[0]
wp_dt = 400
num_steps = num_wp * wp_dt
ts = np.arange(num_steps) * dt

goal = np.zeros((num_steps, n))
K = np.zeros((num_steps,m,6))
for i in range(num_wp):
    goal[i*wp_dt:(i+1)*wp_dt] = waypoints[i]
    K[i*wp_dt:(i+1)*wp_dt] = QF.LQR_tracking_gain(waypoints[i], u_goal, dt, quad)

s_lqr = np.zeros((num_steps, n))
u_lqr = np.zeros((num_steps, m))

for i in trange(1, num_steps):
    state = s_lqr[i-1]
    delta_s = state - goal[i-1]
    u_lqr[i-1] = u_goal + K[i-1] @ delta_s[0:6]
    s_lqr[i] = QF.loaded_dynamics(s_lqr[i-1], u_lqr[i-1], dt, quad)

plt.plot(s_lqr[:,0],s_lqr[:,1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Loaded Dynamics SCP Trajectory")
plt.show()

fig, anim = animate_2D_quad(ts, s_lqr, full_system=0, frameDelay=1)
#display.display(display.HTML(anim.to_html5_video()))
f = r"c://Users/alexe/Desktop/planar_quad_2.gif"
writergif = animation.PillowWriter(fps=30)
writervideo = animation.FFMpegWriter(fps=60)
# anim.save(f, writer=writergif)
plt.show()

plt.plot(ts,u_lqr[:,0])
plt.plot(ts,u_lqr[:,1])
plt.xlabel("Time (s)")
plt.ylabel("Propeller Thrust")
plt.legend(('T1','T2'))
plt.show()
