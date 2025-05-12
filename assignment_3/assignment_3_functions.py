import numpy as np
from scipy.integrate import solve_ivp

def complex_function2(xi = -2, xf = 2, yi = -2, yf = 2, z0 = 0, steps = 500, max_n = 50):
    """ 
    ]This function iterates through a complex equation and for each complex number,
    it keeps track of how many iterations before it diverged (if it did)
      Parameters:
        xi (float): Minimum value for the real axis
        xf (float): Maximum value for the real axis
        yi (float): Minimum value for the imaginary axis
        yf (float): Maximum value for the imaginary axis
        z0 (complex): Initial value of z (default is 0)
        steps (int): number of points in between the values
        max_n (int): Maximum number of iterations allowed before saying unbound
          
        Returns:
        A 2D array with number of iteratiosn before divergeing, or -1 if it does not diverge

          """
    xs = np.linspace(xi, xf, steps)
    ys = np.linspace(yi, yf, steps)
    x, y = np.meshgrid(xs, ys)

    C = x + 1j * y
    iter_counts = np.full(C.shape, -1, dtype=int)  

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            z = z0
            c = C[i, j]
            for n in range(max_n):
                z = z ** 2 + c
                if abs(z) > 2:
                    iter_counts[i, j] = n  
                    break

    return iter_counts

def lorentz_sys(t, funcs, sigma, r, b):
    """
    This function defines the Lorenz system of differential equations, which models atmosphere

    Parameters:

    t (float): Time variable (required by ODE solvers, not used in equation)
    funs (list or array): Current values of the state variables [X, Y, Z]
    sigma (float): Prandtl number
    r (float): Rayleigh number
    b (float): dimensionless length parameter

    Returns:
    derivatives (list): List of three derivatives [dX/dt, dY/dt, dZ/dt] at time t

    """
    X, Y, Z = funcs
    dx_dt = -sigma * (X - Y)
    dy_dt = r * X - Y - X * Z
    dz_dt = -b * Z + X * Y

    return [dx_dt, dy_dt, dz_dt]

def lorentz_ivp_solver(lorentz_system, W0, ti, tf):
    """
    This function uses an ODE solver using scipy to solve the Lorenz system

    Parameters:
        lorentz_system (function): Returns the derivatives for the Lorenz system
        W0 (list): Initial condition [X0, Y0, Z0]
        ti (float): Start time of integration
        tf (float): End time of integration

    Returns:
        sol: A solution object from scipy.integrate.solve_ivp.

    """
        
    t_eval = np.linspace(ti, tf, int((tf - ti) / 0.01))
    sol = solve_ivp(lorentz_system, (ti, tf), y0 = W0, args = (10., 28., 8./3), dense_output = True, t_eval = t_eval)
    
    return sol














