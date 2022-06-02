# Python packages
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
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
s_goal1 = np.array([5., 5., 0., 0., 0., 0., 0., 0.])   # desired location w/ hover final velocity
u_goal = np.array([((quad.m_Q+quad.m_p)*quad.g/2),((quad.m_Q+quad.m_p)*quad.g/2)])    # desired hover control
s_traj1, u_traj1 = QF.generate_scp_trajectory(fd, P, Q, R, N, s_goal1, s0, tol, max_iters, dt, quad)

s_goal2 = np.array([10., 0., 0., 0., 0., 0., 0., 0.]) 
t2 = np.arange(t1[-1], t1[-1] + t_final + dt, dt)
s_traj2, u_traj2 = QF.generate_scp_trajectory(fd, P, Q, R, N, s_goal2, s_traj1[-1,:], tol, max_iters, dt, quad)

t_tot = np.concatenate((t1,t2))
s_traj_tot = np.concatenate((s_traj1, s_traj2))
u_traj_tot = np.concatenate((u_traj1, u_traj2))

s_tot = np.zeros((len(t_tot)+1,n))
s_tot[0] = s0
for k in range(len(t_tot)):
    s_tot[k+1] = QF.loaded_dynamics(s_tot[k], u_traj_tot[0], dt, quad)

plt.plot(s_traj_tot[:,0],s_traj_tot[:,1],label="SCP")
#plt.plot(s_tot[:,0],s_tot[:,1],label = "True")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Loaded Dynamics SCP Trajectory")
plt.legend()
plt.show()

plt.plot(t1,s_traj1[:,3])
plt.plot(t2,s_traj2[:,3])
plt.xlabel("Time (s)")
plt.ylabel("Phi")
plt.title("Pendulum Angle")
plt.show()

plt.plot(t1[:-1],u_traj1[:,0])
plt.plot(t1[:-1],u_traj2[:,1])
plt.plot(t2[:-1],u_traj2[:,0])
plt.plot(t2[:-1],u_traj2[:,1])
plt.xlabel("Time (s)")
plt.ylabel("Propeller Thrust")
plt.legend(('T1','T2'))
plt.show()


fig, anim = animate_2D_quad(t_tot, s_traj_tot, frameDelay=1)
#display.display(display.HTML(anim.to_html5_video()))
f = r"C:\Users\night\Documents\AA203_Final_Project/planar_quad_2.gif"
writergif = animation.PillowWriter(fps=30)
writervideo = animation.FFMpegWriter(fps=60)
# anim.save(f, writer=writergif)
plt.show()


# MPC optimization parameters
N_mpc = 10       # MPC time-step horizon
rf = 1       # final state condition
Q_mpc = np.eye(n)               # state cost matrix
R_mpc = (1/Tmax**2)*np.eye(m)   # control cost matrix
U = np.array([quad.min_thrust_per_prop, quad.max_thrust_per_prop])  # Control bounds
u0 = np.array([(quad.m_Q*quad.g/2),(quad.m_Q*quad.g/2)])            # Initial hover control assuming level attitude

s_mpc1, u_mpc1 = QF.generate_mpc_trajectory(fd, Q_mpc, R_mpc, s0, u0, s_traj1, u_traj1, s_goal1, u_goal, t1, N_mpc, dt, quad, tol, max_iters)
s_mpc2, u_mpc2 = QF.generate_mpc_trajectory(fd, Q_mpc, R_mpc, s_mpc1[-1], u_mpc1[-1], s_traj2, u_traj2, s_goal2, u_goal, t2, N_mpc, dt, quad, tol, max_iters)
s_mpc_tot = np.concatenate((s_mpc1,s_mpc2))
u_mpc_tot = np.concatenate((u_mpc1,u_mpc2))

plt.plot(s_mpc_tot[:,0],s_mpc_tot[:,1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Loaded Dynamics MPC Trajectory")
plt.legend()
plt.grid()
plt.show()

plt.plot(t_tot,u_mpc_tot[:-2,0])
plt.plot(t_tot,u_mpc_tot[:-2,1])
plt.xlabel("Time (s)")
plt.ylabel("Propeller Thrust")
plt.legend(('T1','T2'))
plt.grid()
plt.show()

plt.plot(t_tot,s_mpc_tot[:-2,3])
plt.plot(t_tot,s_mpc_tot[:-2,3])
plt.xlabel("Time (s)")
plt.ylabel("Phi")
plt.title("Pendulum Angle")
plt.show()

fig, anim = animate_2D_quad(t_tot, s_mpc_tot, frameDelay=1)
#display.display(display.HTML(anim.to_html5_video()))
f = r"C:\Users\night\Documents\AA203_Final_Project/planar_quad_2.gif"
writergif = animation.PillowWriter(fps=30)
writervideo = animation.FFMpegWriter(fps=60)
# anim.save(f, writer=writergif)
plt.show()

'''
# Initialize arrays of true state and control trajectories
s = []
u = []
s.append(np.copy(s0))
u.append(np.copy(u0))

# Time discretized mpc trajectories
s_mpc = np.zeros((len(t1)+1, N + 1, n))
u_mpc = np.zeros((len(t1)+1, N, m))

# Initialize MPC trajectories with first SCP converged trajectory
s_warm = s_traj1[0:N+1]
u_warm = u_traj1[0:N]

# perform MPC iterations with warm start
prog_bar = tqdm(range(len(t1)))
for k in prog_bar:
    # Update state and control trajectory for window k based on warm start from previous window
    s_mpc[k], u_mpc[k] = QF.mpc_iteration(fd, Q_mpc, R_mpc, N, s[k], s_goal1, u_goal1, s_warm, u_warm, rf, U, dt, quad, tol, max_iters)

    # Update state and control for one time step using mpc control
    s.append(fd(s[k], u_mpc[k,0,:], dt, quad))
    u.append(u_mpc[k,0,:])

    # Update warm start trajectories
    s_warm[0:N] = s_mpc[k][1:N+1]
    s_warm[-1] = s_mpc[k][-1]
    u_warm[0:N-1] = u_mpc[k][1:N]
    u_warm[N-1] = u_mpc[k][-1]

s = np.array(s)
u = np.array(u)

plt.plot(s[:,0],s[:,1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Loaded Dynamics MPC Trajectory")
plt.legend()
plt.show()

plt.plot(t1,u[:-1,0])
plt.plot(t1,u[:-1,1])
plt.xlabel("Time (s)")
plt.ylabel("Propeller Thrust")
plt.legend(('T1','T2'))
plt.show()
'''

