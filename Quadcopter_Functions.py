# Python Packages
import numpy as np
import cvxpy as cvx
from tqdm import tqdm
from scipy.stats import multivariate_normal
from copy import deepcopy

# unloaded 2D quadrotor object 
class PlanarQuadrotor:
    def __init__(self):

        # Quadrotor parameters
        self.m_Q = 2.     # quadrotor mass (kg)
        self.m_p = 0.5
        self.Iyy = 0.01   # moment of inertia about the out-of-plane axis (kg * m**2)
        self.d = 0.25     # length from center of mass to propellers (m)
        self.g = 9.81     # acceleration due to gravity [m/s^2]
        self.l = 1        # pendulum length (m)
        
        # Control constraints
        self.max_thrust_per_prop = self.m_Q * self.g  # total thrust-to-weight ratio = 1.5
        self.min_thrust_per_prop = 0.

def LoadedQuadEOM(s: np.ndarray, u: np.ndarray, quad):
    # List of states- hello
    # 0 - x
    # 1 - z nice to meet you
    # 2 - theta
    # 3 - phi
    # 4 - xdot
    # 5 - zdot
    # 6 - thetadot
    # 7 - phidot

    # control input is the left and right propeller thrusts
    m_Q, m_p, I_yy, d, g, l = quad.m_Q, quad.m_p, quad.Iyy, quad.d, quad.g, quad.l
    m_tot = m_Q + m_p

    x, z, theta, phi = s[0], s[1], s[2], s[3]
    xdot, zdot, thetadot, phidot = s[4], s[5], s[6], s[7]
    T1, T2 = u
    C_d = 2

    sdot = np.zeros((s.shape[0]))
    sdot[0] = xdot
    sdot[1] = zdot
    sdot[2] = thetadot
    sdot[3] = phidot
    sdot[4] = (T1+T2) * np.sin(theta) * (m_Q + m_p * (np.cos(phi)) ** 2) / (m_Q * m_tot) + (T1+T2) * np.cos(theta) * (m_p * np.sin(phi) * np.cos(phi)) / (m_Q * m_tot) + m_p * l * (phidot) ** 2 * np.sin(phi) / m_tot
    sdot[5] = (T1+T2) * np.cos(theta) * (m_Q + m_p * (np.sin(phi)) ** 2) / (m_Q * m_tot) + (T1+T2) * np.sin(theta) * (m_p * np.sin(phi) * np.cos(phi)) / (m_Q * m_tot) - m_p * l * (phidot) ** 2 * np.cos(phi) / m_tot - g
    sdot[6] = d*(T1-T2) / I_yy - C_d*(thetadot - phidot)
    sdot[7] = -(T1+T2) * np.sin(phi - theta) / (m_Q * l) - C_d*(phidot - thetadot)

    return sdot

def loaded_dynamics(s: np.ndarray, u: np.ndarray, dt: float, quad):
    """Continuous-time dynamics of loaded planar quadrotor expressed as an Euler integration."""
    
    ds = LoadedQuadEOM(s, u, quad)

    return s + dt * ds

def Jacobians(fd: callable, s: np.ndarray, u: np.ndarray, dt: float, quad):
    '''Accept vector of states and control, and output time discretized jacobian matrices'''
    m_Q, m_p, I_yy, l, d = quad.m_Q, quad.m_p, quad.Iyy, quad.l, quad.d
    state_dim = 8
    control_dim = 2
    A_k, B_k, c_k = [], [], []

    C_d = 2
    for k in range(s.shape[0]):
        s_k = s[k]
        u_k = u[k]
        x, z, theta, phi, xdot, zdot, thetadot, phidot = s_k
        T1, T2 = u_k
        
        C1 = (m_Q + m_p)*(T1+T2)/(m_Q*(m_Q + m_p))
        C2 = (m_p)*(T1+T2)/(m_Q*(m_Q + m_p))
        C3 = m_p*l/(m_Q + m_p)
        C4 = (T1-T2)/I_yy
        C5 = (T1+T2)/(m_Q*l)
        
        # State Jacobian
        A = np.eye(state_dim)
        A[0,4] = dt
        A[1,5] = dt
        A[2,6] = dt
        A[3,7] = dt
        A[4,2] = dt*((m_Q + m_p*(np.cos(phi)**2)*(T1+T2))/(m_Q*(m_Q + m_p)) * np.cos(theta) - C2*np.sin(phi)*np.cos(phi)*np.sin(theta))
        A[4,3] = dt*((-2*m_p*np.cos(phi)*np.sin(phi))*(T1+T2)/(m_Q*(m_Q + m_p)) * np.sin(theta) + C2*np.cos(2*phi)*np.cos(theta) + C3*(phidot**2)*np.sin(phi))
        A[4,7] = dt*(-C3*(2*phidot*np.cos(phi)))      
        A[5,2] = dt*(-(m_Q + m_p*((np.sin(phi))**2))*(T1+T2)/(m_Q*(m_Q + m_p))*np.sin(theta) + C2*np.sin(phi)*np.cos(phi)*np.cos(theta))
        A[5,3] = dt*((-m_p*np.sin(2*phi))*(T1+T2)/(m_Q*(m_Q + m_p))*np.cos(theta) + C2*np.cos(2*phi)*np.sin(theta) + C3*(phidot**2)*np.sin(phi))
        A[5,7] = dt*(-(C3*2*phidot)*np.cos(phi))
        A[6,6] = -C_d*dt
        A[6,7] = C_d*dt
        A[7,2] = dt*(C5*np.cos(phi - theta))
        A[7,3] = dt*(-C5*np.cos(phi - theta))
        A[7,6] = C_d*dt
        A[7,7] = -C_d*dt

        U1 = (m_Q + m_p*(np.cos(phi)**2))*np.sin(theta)/(m_Q*(m_Q + m_p))
        U2 = (m_p)*(np.sin(phi)*np.cos(phi)*np.cos(theta))/(m_Q*(m_Q + m_p))
        U3 = (m_Q + m_p*(np.sin(phi)**2))*np.cos(theta)/(m_Q*(m_Q + m_p))
        U4 = (m_p)*(np.sin(phi)*np.cos(phi)*np.sin(theta))/(m_Q*(m_Q + m_p))
        U5 = d/I_yy
        U6 = -np.sin(phi-theta)/(m_Q*l)

        # Measurement Jacobian
        B = np.zeros((state_dim, control_dim))
        B[4,0] = dt*(U1 + U2)
        B[4,1] = dt*(U1 + U2)
        B[5,0] = dt*(U3 + U4)
        B[5,1] = dt*(U3 + U4)
        B[6,0] = dt*U5
        B[6,1] = -dt*U5
        B[7,0] = U6
        B[7,1] = U6

        A_k.append(A)
        B_k.append(B)
        # Additional linearization constant
        c_k.append(fd(s_k, u_k, dt, quad) - A @ s_k - B @ u_k)

    return A_k, B_k, c_k

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
def generate_scp_trajectory(fd: callable, P: np.ndarray, Q: np.ndarray, R: np.ndarray, N: int, 
                            s_goal: np.ndarray, s0: np.ndarray, tol: float, max_iters: int, dt: float, quad):
    '''Solve the quadrotor trajectory problem using SCP'''
    n = Q.shape[0]    # state dimension
    m = R.shape[0]    # control dimension

    # Initialize nominal (zero control) trajectories
    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = fd(s_bar[k], u_bar[k], dt, quad)
    
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
    A, B, c = Jacobians(fd, s_bar[:-1], u_bar, dt, quad)
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
        constraints += [s_cvx[k+1] == A[k]@s_cvx[k] + B[k]@u_cvx[k] + c[k]]             # linearized dynamics constraint
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

# Compute deviation variable gain matrix
def LQR_tracking_gain(s_goal: np.ndarray, u_goal: np.ndarray, dt: float, quad):
    # Deviation cost matrices
    Q_LQR = 1e-3*np.eye(6)  # state deviation cost matrix
    R_LQR = 1e-3*np.eye(2)  # control deviation cost matrix

    # Initialize cost-to-go matrix to 0
    P_inf = np.zeros_like(Q_LQR)

    A, B = LQR_Jacobians(s_goal[0:6], u_goal[0:6], dt, quad)
    # P_next stores P_{k+1} matrix
    P_next = deepcopy(P_inf)
    not_converged = True
    while not_converged:
        # Ricatti Recursion update step
        K = -1 * np.linalg.inv(R_LQR + B.T @ P_next @ B) @ B.T @ P_next @ A
        P = Q_LQR + A.T @ P_next @ (A + B @ K)

        # maximum element-wise norm condition ||P_k+1 - P_k||_max < 1e-4
        if np.all(np.absolute(P_next - P) < 1e-5):
            not_converged = False

        # Update cost-to-go matrix for next loop
        P_next = deepcopy(P)

    # Infinite horizon deviation variable gain matrix
    return K