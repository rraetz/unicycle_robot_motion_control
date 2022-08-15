# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 09:01:39 2022

@author: Raphael
"""

#%% Imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



#%% Functions
def rho_delta(xm, ym, theta, xr, yr):
    rho = np.sqrt((xr-xm)**2 + (yr-ym)**2)
    delta = np.arctan2((yr-ym),(xr-xm)) - theta  
    return rho, delta

def limit(val, min_val, max_val):
    return np.minimum(max_val, np.maximum(val, min_val))

def controller(rho, delta, k1, k2):
    v = k1*np.cos(delta)*rho
    v_max = 10
    v_limited = limit(v, -v_max, v_max)
    if (np.abs(v) > 0):
        k_comp = v_limited/v
    else: 
        k_comp = 1        
    omega = k1*k2*np.sin(delta)*np.cos(delta)*k_comp
    return v_limited, omega


def kinematics(v, theta, omega):
    xm_d = np.cos(theta)*v
    ym_d = np.sin(theta)*v
    theta_d = omega
    return xm_d, ym_d, theta_d


def dy_dt(t, y, xr, yr, k1, k2):
    xm, ym, theta = y
    v, omega = controller(*rho_delta(xm, ym, theta, xr, yr), k1, k2)
    xm_d, ym_d, theta_d =  kinematics(v, theta, omega) 
    return np.hstack((xm_d, ym_d, theta_d))


#%% Solve
x0 = (0, 10, -20, 30, -40, 50)
y0 = (0, 0, -10, 50, 30, 10)
theta0 = (0, 1, 2, 3, 4, 5)
xr = 20; yr = 10
k1 = 1; k2 = 2

n_eval = 1000
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], n_eval+1)

for i in range(0, len(x0)):  
    sol = solve_ivp(
        dy_dt, 
        t_span, 
        np.array((x0[i], y0[i], theta0[i])) , 
        t_eval=t_eval, 
        args=(xr, yr, k1, k2))

    # Plot
    x = sol.y[0,:].T
    y = sol.y[1,:].T
    theta = sol.y[2,:].T
    plt.plot(x,y)
    axes=plt.gca()
    axes.set_aspect(1)
    axes.grid('on')
    
axes = plt.gca()
axes.set_ylabel('y')
axes.set_xlabel('x')






