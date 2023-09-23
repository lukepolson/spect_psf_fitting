import numpy as np
from scipy.optimize import minimize


def dual_exponential(d, b1, b2, b3, b4):
    return b1*np.exp(-b2*d)+b3*np.exp(-b4*d)

def sqrt_fit(d, b1, b2, b3):
    return np.sqrt(b1*d**2+b2)+b3

def gaus(x, y, params):
    mux, muy, sigma, A = params
    return A*np.exp(-(x-mux)**2 / (2*sigma**2) -(y-muy)**2 / (2*sigma**2))

def initial_gaus_estimate(data, sigma_init=0.5, amplitude_init=0.3):
    N = data.shape[0]
    x_data = y_data =  np.arange(N) + 0.5 - N/2
    x_grid, y_grid = np.meshgrid(x_data, y_data)
    min_function = lambda params: np.sum((gaus(x_grid, y_grid, params)-data)**2)
    return minimize(min_function, x0=np.array([0,0,sigma_init,amplitude_init])).x