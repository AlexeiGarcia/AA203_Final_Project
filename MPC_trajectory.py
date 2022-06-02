# SCP Trajectory Generation code for unloaded quadcopter dynamics

# Python packages
from re import S
from xmlrpc.client import Boolean
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import solve_discrete_are
import matplotlib.animation as animation
# custom functions
from quad_animation import animate_2D_quad

# unloaded 2D quadrotor object 
class PlanarQuadrotor:
    def __init__(self, m_Q, Iyy, d, m_p = 0.):
        self.m_Q = m_Q      # quadrotor mass
        self.Iyy = Iyy      # quadrotor second moment of inertia
        self.d = d          # length from center of mass to propellers
        self.m_p = m_p      # payload mass (optional)
        self.g = 9.81       # acceleration due to gravity [m/s^2]

        # Control constraints
        self.max_thrust_per_prop = self.m_Q * self.g  # total thrust-to-weight ratio = 1.5
        self.min_thrust_per_prop = 0.

def unloaded_dynamics(unloaded_state, control, quad):
    """Continuous-time dynamics of an unloaded planar quadrotor expressed as an Euler integration."""
    x, z, theta, v_x, v_z, omega = unloaded_state
    T1, T2 = control
    m, Iyy, d, g = quad.m_Q, quad.Iyy, quad.d, quad.g
    ds = np.array([v_x, v_z, omega, ((T1 + T2) * np.sin(theta)) / m, ((T1 + T2) * np.cos(theta)) / m - g, (T1-T2)*d / Iyy])
    return unloaded_state + dt * ds

def linearize(fd: callable, s: np.ndarray, u: np.ndarray, dt: float, quad):
    """Explicitly linearize the unloaded dynamics around (s,u)."""
    state_dim = s.shape[1]
    control_dim = u.shape[1]
    m_Q, Iyy, d = quad.m_Q, quad.Iyy, quad.d
    A, B, c = [], [], []

    for k in range(s.shape[0]):
        s_k = s[k]
        u_k = u[k]
        x, z, theta, v_x, v_y, omega = s_k
        T1, T2 = u_k

        # derivative wrt state variables
        df_ds = np.identity(state_dim)
        df_ds[0:3,3:6] = dt*np.identity(3)
        df_ds[3,2] = dt*(T1+T2)*np.cos(theta) / m_Q
        df_ds[4,2] = -dt*(T1+T2)*np.sin(theta) / m_Q

        # derivative wrt control variables
        df_du = np.zeros((state_dim, control_dim))
        df_du[3,0] = dt*np.sin(theta) / m_Q
        df_du[3,1] = dt*np.sin(theta) / m_Q
        df_du[4,0] = dt*np.cos(theta) / m_Q
        df_du[4,1] = dt*np.cos(theta) / m_Q
        df_du[5,0] = dt*d / Iyy
        df_du[5,1] = -dt*d / Iyy

        A.append(df_ds)
        B.append(df_du)
        c.append(fd(s_k, u_k, quad) - df_ds @ s_k - df_du @ u_k)
 
    return A, B, c

def LQR_Jacobians(s: np.ndarray, u: np.ndarray, dt: float, quad):
    """Calculate state and control Jacobians around (s*,u*)."""
    state_dim = s.shape[0]
    control_dim = u.shape[0]
    m, Iyy, d = quad.m_Q, quad.Iyy, quad.d
    x, z, theta, v_x, v_y, omega = s
    T1, T2 = u

    # derivative wrt state variables
    df_ds = np.identity(state_dim)
    df_ds[0:3,3:6] = dt*np.identity(3)
    df_ds[3,2] = dt*(T1+T2)*np.cos(theta) / m
    df_ds[4,2] = -dt*(T1+T2)*np.sin(theta) / m

    # derivative wrt control variables
    df_du = np.zeros((state_dim, control_dim))
    df_du[3,0] = dt*np.sin(theta) / m
    df_du[3,1] = dt*np.sin(theta) / m
    df_du[4,0] = dt*np.cos(theta) / m
    df_du[4,1] = dt*np.cos(theta) / m
    df_du[5,0] = dt*d / Iyy
    df_du[5,1] = -dt*d / Iyy

    return df_ds, df_du

# Generate time discretized control policy using Sequential Convex Programming
def generate_scp_trajectory(fd: callable, P: np.ndarray, Q: np.ndarray, R: np.ndarray, N: int, s_goal: np.ndarray, 
                            s0: np.ndarray, tol: float, max_iters: int, dt: float, quad):
    '''Solve the quadrotor trajectory problem using SCP'''
    n = Q.shape[0]    # state dimension
    m = R.shape[0]    # control dimension

    # Initialize nominal (zero control) trajectories
    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = fd(s_bar[k], u_bar[k], quad)
    
    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    obj_prev = np.inf
    prog_bar = tqdm(range(max_iters))
    for i in prog_bar:
        s, u, obj = scp_iteration(fd, P, Q, R, N, s_bar, u_bar, s_goal, s0, dt, quad)
        diff_obj = np.abs(obj - obj_prev)
        prog_bar.set_postfix({'objective change': '{:.5f}'.format(diff_obj)})

        if diff_obj < tol:
            converged = True
            print('SCP converged after {} iterations.'.format(i))
            break
        else:
            obj_prev = obj
            np.copyto(s_bar, s)
            np.copyto(u_bar, u)

    if not converged:
        raise RuntimeError('SCP did not converge!')

    return s, u

def scp_iteration(fd: callable, P: np.ndarray, Q: np.ndarray, R: np.ndarray, N: int, s_bar: np.ndarray, 
                  u_bar: np.ndarray, s_goal: np.ndarray, s0: np.ndarray, dt: float, quad):
    """Solve a single SCP sub-problem for the quadrotor trajectory problem."""
    A, B, c = linearize(fd, s_bar[:-1], u_bar, dt, quad)
    A, B, c = np.array(A), np.array(B), np.array(c)
    n = Q.shape[0]
    m = R.shape[0]
    T_min = quad.min_thrust_per_prop
    T_max = quad.max_thrust_per_prop
    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    # ------------------- Cost -------------------
    # terminal state cost
    terminal_cost = cvx.quad_form((s_cvx[-1] - s_goal), P)

    # summed position and control cost
    sum_cost = 0
    for k in range(N):
        sum_cost += cvx.sum(cvx.quad_form((s_cvx[k]-s_goal), Q) + cvx.quad_form(u_cvx[k], R))

    objective = sum_cost + terminal_cost

    # ------------------- Constraints -------------------

    # initial position constraint
    constraints = [s_cvx[0] == s0]

    for k in range(N):
        constraints += [s_cvx[k+1] == A[k]@s_cvx[k] + B[k]@u_cvx[k] + c[k]] # linearized dynamics constraint
        constraints += [cvx.abs(u_cvx[k][0] - ((T_max+T_min)/2)) <= (T_max+T_min)/2]    # Force control bounds
        constraints += [cvx.abs(u_cvx[k][1] - ((T_max+T_min)/2)) <= (T_max+T_min)/2]    # Force control bounds

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve()

    if prob.status != 'optimal':
        raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)

    s = s_cvx.value
    u = u_cvx.value
    obj = prob.objective.value

    return s, u, obj

def mpc_iteration(fd: callable, Q: np.ndarray, R: np.ndarray, N: int, s0: np.ndarray, s_goal: np.ndarray, u_goal: np.ndarray, 
                  s_warm: np.ndarray, u_warm: np.ndarray, rf: float, U: float, dt: float, quad):
    """Solve a single MPC sub-problem for the quadrotor trajectory problem."""
    n = Q.shape[0]
    m = R.shape[0]
    # A, B, c = linearize(fd, s_goal, u_goal, dt, quad)
    # A, B, c = np.array(A), np.array(B), np.array(c)
    A_N, B_N = LQR_Jacobians(s_goal, u_goal, dt, quad)
    A_N, B_N = np.array(A_N), np.array(B_N)
    P = solve_discrete_are(A_N, B_N, Q, R)

    '''
    thrust_min = U[0]
    thrust_max = U[1]
    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    # terminal state cost
    terminal_cost = cvx.quad_form((s_cvx[-1] - s_goal), P)

    # terminal_cost = 1e3 * cvx.norm(s_cvx[-1] - s_goal)

    # summed position and control cost
    sum_cost = 0
    for k in range(N):
        sum_cost += cvx.sum(cvx.quad_form((s_cvx[k]-s_goal), Q) + cvx.quad_form(u_cvx[k], R)) 

    objective = sum_cost + terminal_cost

    constraints = [s_cvx[0] == s0]                                          # initial position constraint
    for k in range(N):
        constraints += [s_cvx[k+1] == A@s_cvx[k] + B@u_cvx[k] + c] # linearized dynamics constraint
        constraints += [cvx.abs(u_cvx[k][0] - ((thrust_max+thrust_min)/2)) <= (thrust_max+thrust_min)/2]    # Force control bounds
        constraints += [cvx.abs(u_cvx[k][1] - ((thrust_max+thrust_min)/2)) <= (thrust_max+thrust_min)/2]    # Force control bounds
    constraints += [cvx.norm_inf(s_cvx[-1]) <= rf]              # terminal state constraint

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve()

    s = s_cvx.value
    u = u_cvx.value
    status = prob.status
    '''

    # Initialize nominal (zero control) trajectories from warm values
    u_bar = u_warm
    s_bar = s_warm
    s_bar[0] = s0
    #for k in range(N):
    #    s_bar[k+1] = fd(s_bar[k], u_bar[k], quad)

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    obj_prev = np.inf
    for i in range(max_iters):
        s, u, obj = scp_iteration(fd, P, Q, R, N, s_bar, u_bar, s_goal, s0, dt, quad)
        diff_obj = np.abs(obj - obj_prev)
        prog_bar.set_postfix({'objective change': '{:.5f}'.format(diff_obj)})

        if diff_obj < tol:
            converged = True
            # print('SCP converged after {} iterations.'.format(i))
            break
        else:
            obj_prev = obj
            np.copyto(s_bar, s)
            np.copyto(u_bar, u)

    if not converged:
        raise RuntimeError('SCP did not converge!')

    return s, u


# Define planar quadrotor object
m_Q = 2.    # quadrotor mass (kg)
Iyy = 0.01  # moment of inertia about the out-of-plane axis (kg * m**2)
d = 0.25    # half-length (m)
quad = PlanarQuadrotor(m_Q, Iyy, d)

# quadrotor trajectory optimization simulation parameters
dt = 0.01                                                   # Simulation time step - sec
t_final = 5.                                                # Simulation run time - sec
t = np.arange(0., t_final + dt, dt)                         # time array
n = 6                                                       # state dimension
m = 2                                                       # control dimension
s0 = np.array([0., 0., 0., 0., 0., 0.])                     # initial level state
u0 = np.array([(quad.m_Q*quad.g/2),(quad.m_Q*quad.g/2)])    # Initial hover control assuming level attitude
s_goal = np.array([3., 1., 0., 0., 0., 0.])                 # desired location w/ hover final velocity
u_goal = np.array([(quad.m_Q*quad.g/2),(quad.m_Q*quad.g/2)]) # Target hover control

# Control bounds
Tmin = quad.min_thrust_per_prop
Tmax = quad.max_thrust_per_prop

# SCP parameters
P_scp = 150*np.eye(n)   # terminal state cost matrix
Q_scp = np.eye(n)       # state cost matrix
R_scp = (1/Tmax**2)*np.eye(m)  # control cost matrix
tol = 0.5          # convergence tolerance
max_iters = 100     # maximum number of SCP iterations

# Dynamics propagation function (callable)
fd = unloaded_dynamics

# Generate the initial trajectory with SCP
N_scp = t.size - 1
s_scp, u_scp = generate_scp_trajectory(fd, P_scp, Q_scp, R_scp, N_scp, s_goal, s0, tol, max_iters, dt, quad)

# Plot initial SCP trajectory
plt.plot(s_scp[:,0],s_scp[:,1])
plt.title('Initial SCP Quadrotor Trajectory')
plt.xlabel('X - position (m)')
plt.ylabel('Z - position (m)')
plt.grid()
plt.show()

plt.plot(t[:-1],u_scp[:,0])
plt.plot(t[:-1],u_scp[:,1])
plt.title('SCP Control v Time')
plt.xlabel('Time (s)')
plt.ylabel('Thrust')
plt.legend(("Thrust 1","Thrust 2"))
plt.grid()
plt.show()

fig, anim = animate_2D_quad(t, s_scp, full_system=0, frameDelay=1)
#display.display(display.HTML(anim.to_html5_video()))
f = r"c://Users/alexe/Desktop/planar_quad_scp.gif"
writergif = animation.PillowWriter(fps=30)
writervideo = animation.FFMpegWriter(fps=60)
# anim.save(f, writer=writergif)
plt.show()

# MPC optimization parameters
N = 10       # MPC time-step horizon
rf = 1       # final state condition
Q_mpc = np.eye(n)               # state cost matrix
R_mpc = (1/Tmax**2)*np.eye(m)   # control cost matrix
U = np.array([quad.min_thrust_per_prop, quad.max_thrust_per_prop])  # Control bounds

# Initialize arrays of true state and control trajectories
s = []
u = []
s.append(np.copy(s0))
u.append(np.copy(u0))

# Time discretized mpc trajectories
s_mpc = np.zeros((len(t)+1, N + 1, n))
u_mpc = np.zeros((len(t)+1, N, m))

# Initialize MPC trajectories with first SCP converged trajectory
s_warm = s_scp[0:N+1]
u_warm = u_scp[0:N]

# perform MPC iterations with warm start
prog_bar = tqdm(range(len(t)))
for k in prog_bar:
    # Update state and control trajectory for window k based on warm start from previous window
    s_mpc[k], u_mpc[k] = mpc_iteration(fd, Q_mpc, R_mpc, N, s[k], s_goal, u_goal, s_warm, u_warm, rf, U, dt, quad)
    '''
    if status == 'infeasible':
        s_mpc = s_mpc[:k]
        u_mpc = u_mpc[:k]
        break
    '''

    # Update state and control for one time step using mpc control
    s.append(fd(s[k], u_mpc[k,0,:], quad))
    u.append(u_mpc[k,0,:])

    # Update warm start trajectories
    s_warm[0:N] = s_mpc[k][1:N+1]
    s_warm[-1] = s_mpc[k][-1]
    u_warm[0:N-1] = u_mpc[k][1:N]
    u_warm[N-1] = u_mpc[k][-1]

s = np.array(s)
u = np.array(u)

# Plot MPC
plt.plot(s[:,0],s[:,1])
plt.title('MPC Quadrotor Trajectory')
plt.xlabel('X - position (m)')
plt.ylabel('Z - position (m)')
plt.grid()
plt.show()

plt.plot(t,u[:-1,0])
plt.plot(t,u[:-1,1])
plt.title('MPC Control v Time')
plt.xlabel('Time (s)')
plt.ylabel('Thrust')
plt.legend(("Thrust 1","Thrust 2"))
plt.grid()
plt.show()

fig, anim = animate_2D_quad(t, s, full_system=0, frameDelay=1)
#display.display(display.HTML(anim.to_html5_video()))
f = r"c://Users/alexe/Desktop/planar_quad_MCP.gif"
writergif = animation.PillowWriter(fps=30)
writervideo = animation.FFMpegWriter(fps=60)
# anim.save(f, writer=writergif)
plt.show()