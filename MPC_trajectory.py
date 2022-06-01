# SCP Trajectory Generation code for unloaded quadcopter dynamics

# Python packages
from re import S
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import solve_discrete_are

# unloaded 2D quadrotor object 
class PlanarQuadrotor:
    def __init__(self, m_Q, Iyy, l, m_p = 0.):
        self.m_Q = m_Q      # quadrotor mass
        self.Iyy = Iyy      # quadrotor second moment of inertia
        self.l = l          # length from center of mass to propellers
        self.m_p = m_p      # payload mass (optional)
        self.g = 9.81       # acceleration due to gravity [m/s^2]

        # Control constraints
        self.max_thrust_per_prop = self.m_Q * self.g  # total thrust-to-weight ratio = 1.5
        self.min_thrust_per_prop = 0.

def unloaded_dynamics(unloaded_state, control, quad):
    """Continuous-time dynamics of an unloaded planar quadrotor expressed as an Euler integration."""
    x, z, theta, v_x, v_z, omega = unloaded_state
    T1, T2 = control
    m, Iyy, l, g = quad.m_Q, quad.Iyy, quad.l, quad.g
    ds = np.array([v_x, v_z, omega, ((T1 + T2) * np.sin(theta)) / m, ((T1 + T2) * np.cos(theta)) / m - g, (T1-T2)*l / Iyy])
    return unloaded_state + dt * ds

def linearize(fd: callable, s: np.ndarray, u: np.ndarray, dt: float, quad):
    """Explicitly linearize the unloaded dynamics around (s,u)."""
    state_dim = s.shape[0]
    control_dim = u.shape[0]
    m, Iyy, l = quad.m_Q, quad.Iyy, quad.l

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
    df_du[5,0] = dt*l / Iyy
    df_du[5,1] = -dt*l / Iyy

    A = df_ds
    B = df_du
    c = fd(s, u, quad) - df_ds @ s - df_du @ u
 
    return A, B, c

def mpc_iteration(fd: callable, Q: np.ndarray, R: np.ndarray, N: int, s_goal: np.ndarray, u_goal: np.ndarray, 
                  s0: np.ndarray, u0: np.ndarray, rf: float, U: float, dt: float, quad):
    """Solve a single MPC sub-problem for the quadrotor trajectory problem."""
    n = Q.shape[0]
    m = R.shape[0]
    A, B, c = linearize(fd, s_goal, u_goal, dt, quad)
    # A, B, c = linearize(fd, s0, u0, dt, quad)
    # A, B, c = np.array(A), np.array(B), np.array(c)
    A_N, B_N, _ = linearize(fd, s_goal, u_goal, dt, quad)
    A_N, B_N = np.array(A_N), np.array(B_N)
    P = solve_discrete_are(A_N, B_N, Q, R)
    # P = 1e3*np.eye(n)
    # P = np.diag([1e3, 1e3, 1e3, 1., 1., 1.])

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

    return s, u, status

# Define planar quadrotor object
m_Q = 2     # quadrotor mass (kg)
Iyy = 0.01  # moment of inertia about the out-of-plane axis (kg * m**2)
l = 0.25    # half-length (m)
quad = PlanarQuadrotor(m_Q, Iyy, l)

# quadrotor trajectory optimization parameters
dt = 0.01                                                   # Simulation time step - sec
t_final = 5.                                                # Simulation run time - sec
n = 6                                                       # state dimension
m = 2                                                       # control dimension
s0 = np.array([0., 0., 0., 0., 0., 0.])                     # initial level state
#u0 = np.array([(quad.m_Q*quad.g/2),(quad.m_Q*quad.g/2)])   # Initial hover control assuming level attitude
u0 = np.array([0.,0.])
s_goal = np.array([1., 1., 0., 0., 0., 0.])                 # desired location w/ hover final velocity
# Target hover control
u_goal = np.array([(quad.m_Q*quad.g/2),(quad.m_Q*quad.g/2)])

# Control bounds
U = np.array([quad.min_thrust_per_prop, quad.max_thrust_per_prop])

# SCP parameters
Q = np.eye(n)
R = (1/quad.max_thrust_per_prop**2)*np.eye(m)       # control cost matrix

# Dynamics propagation
fd = unloaded_dynamics

# Solve the swing-up problem with MPC
t = np.arange(0., t_final + dt, dt)
N = 5       # MPC time-step horizon
rf = 1      # final state condition

s = []
u = []
s.append(np.copy(s0))
u.append(np.copy(u0))
s_mpc = np.zeros((len(t), N + 1, n))
u_mpc = np.zeros((len(t), N, m))
for k in range(len(t)):
    s_mpc[k], u_mpc[k], status = mpc_iteration(fd, Q, R, N, s_goal, u_goal, s[k], u[k], rf, U, dt, quad)
    if status == 'infeasible':
        s_mpc = s_mpc[:k]
        u_mpc = u_mpc[:k]
        break
    s.append(fd(s[k], u_mpc[k,0,:], quad))
    u.append(u_mpc[k,0,:])
s = np.array(s)
u = np.array(u)

# Plot and Animate quadrotor
plt.plot(s[:,0],s[:,1])
plt.title('Quadrotor Trajectory')
plt.xlabel('X - position (m)')
plt.ylabel('Z - position (m)')
plt.grid()
plt.show()

plt.plot(t,u[:-1,0])
plt.plot(t,u[:-1,1])
plt.title('Control v Time')
plt.xlabel('Time (s)')
plt.ylabel('Thrust')
plt.legend(("Thrust 1","Thrust 2"))
plt.grid()
plt.show()

