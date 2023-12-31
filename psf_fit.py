import numpy as np
from copy import deepcopy
import numpy as np
from scipy.interpolate import bisplev
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from fit_part1 import dual_exponential, sqrt_fit, gaus, initial_gaus_estimate
from loss_functions import negativity_loss
from load_data import get_projections_spacing_radius
from scipy.integrate import dblquad

# Unfortunately distances have to be provided seperately because not stored in STATIC SPECT scan .h00 files (single projection)
def get_psf_kernel(projection_paths):
    psfs_data = []
    distances = []
    for projection_path in projection_paths:
        projections, dr, distance = get_projections_spacing_radius(projection_path)
        psfs_data.append(projections)
        distances.append(distance)
    return fit(np.array(psfs_data), np.array(distances), dr)[0]

# Kernel function: note, when returned by "fit", only takes in x, y, and d; rest of the arguments given as default based on the fit
def kernel(
    x,
    y,
    d,
    dr,
    gaus_sigma_fit,
    gaus_amplitude_fit,
    spline_sigma_fit,
    spline_amplitude_fit,
    min_spline_sigma,
    tck,
    x_data,
    y_data,
    L,
    gaus_only=False,
    spline_only=False,
    normalize=True):
        x_new = x/dr[0] # in units of voxels corresponding to original fit
        y_new = y/dr[1] # in units of voxels corresponding to original fit
        x_grid, y_grid = np.meshgrid(x_new, y_new)
        gaus_sigma = sqrt_fit(d, *gaus_sigma_fit)
        gaus_amplitude = dual_exponential(d, *gaus_amplitude_fit)
        gaus_comp = gaus(x_grid, y_grid, [0, 0, gaus_sigma, gaus_amplitude])
        spline_sigma = sqrt_fit(d, *spline_sigma_fit)
        spline_amplitude = dual_exponential(d, *spline_amplitude_fit)
        spline_comp = spline_amplitude*bisplev(x_new/spline_sigma, y_new/spline_sigma, tck)
        C = spline_sigma/min_spline_sigma
        spline_comp[(x_grid>C*x_data.max())+(x_grid<C*x_data.min())+(y_grid>C*y_data.max())+(y_grid<C*y_data.min())] = 0 # set outside fitted boundaries equal to zero
        if gaus_only:
            k = gaus_comp
        elif spline_only:
            k = spline_comp
        else:
            k = gaus_comp + spline_comp
        # potentially normalize
        if normalize:
            dx = np.diff(x_new)[0]
            dy = np.diff(y_new)[0]
            N = gaus_amplitude * (2*np.pi) * (gaus_sigma/dx)*(gaus_sigma/dy)  + spline_amplitude * L *(spline_sigma/dx)*(spline_sigma/dy)
            k = k/N
        return k
            
        
def fit(psfs_data, distances, dr=(1,1), spline_order=3, N_points=64, negativity_loss_scaling=20):
    N = psfs_data.shape[0]
    Nx = psfs_data.shape[1]
    x_data = y_data =  np.arange(Nx) + 0.5 - Nx/2
    x_grid, y_grid = np.meshgrid(x_data, y_data)
    # Set up inital spline (larger than x_data in case of scaling outwards)
    x_min = 1.5*min(x_data)
    x_max = 1.5*max(x_data)
    tx = ty = np.concatenate([np.ones(spline_order)*x_min,
                              np.linspace(x_min,x_max,N_points),
                              np.ones(spline_order)*x_max])
    cc_init = np.zeros((N_points+spline_order-1)**2)+1e-3
    tck = [tx, ty, cc_init, spline_order, spline_order]
    # Objective function that takes in Gaussian parameters and spline knots
    def loss_function_gaus_and_splineknots(params):
        tck_curr = deepcopy(tck)
        gg = params[:4*N].reshape((N,-1))
        bh = params[4*N:5*N]
        bs = params[5*N:6*N]
        tck_curr[2] = params[6*N:]
        spline_component = np.array([bh_i*bisplev(x_data/bs_i, y_data/bs_i, tck_curr) for bh_i, bs_i in zip(bh, bs)])
        gaus_component = np.array([gaus(x_grid, y_grid, gg_i) for gg_i in gg])
        fit =  spline_component + gaus_component
        return np.sum((psfs_data - fit) ** 2) +negativity_loss_scaling*negativity_loss(spline_component)
    # Objective function that takes in spline amplitude/width scaling
    def loss_function_spline_scaling(spline_scaling, spline_tcks, gaus_params):
        tck_curr = deepcopy(tck)
        gg = gaus_params.reshape((N,-1))
        bh = spline_scaling[:N]
        bs = spline_scaling[N:]
        spline_tcks[spline_tcks<=0] = 0
        tck_curr[2] = spline_tcks
        spline_component = np.array([bh_i*bisplev(x_data/bs_i, y_data/bs_i, tck_curr) for bh_i, bs_i in zip(bh, bs)])
        gaus_component = np.array([gaus(x_grid, y_grid, gg_i) for gg_i in gg])
        fit =  spline_component + gaus_component
        return np.sum((psfs_data - fit) ** 2) + negativity_loss_scaling*negativity_loss(spline_component)
    gaus_params_init = np.concatenate([initial_gaus_estimate(psf_data)  for psf_data in psfs_data])
    bspline_heights_init = np.ones(N)
    bspline_sigmas_init = np.ones(N)
    bspline_params_init = tck[2]
    # Initialize optimal params
    params_opt = np.concatenate([gaus_params_init, bspline_heights_init, bspline_sigmas_init, bspline_params_init])
    # Alternate optimizing for gaussian+spline tcks and spline scaling (needs to be done seperate for some reason: othewise spline scaling doesnt change)
    for i in range(5):
        bounds = 6*N*[(-np.inf, np.inf)] + len(params_opt[6*N:])*[(0,np.inf)]
        params_opt = minimize(loss_function_gaus_and_splineknots, params_opt, method='L-BFGS-B', bounds=bounds).x
        # Nelder-Mead method seems more stable for this one
        params_spline_scaling_new_opt = minimize(loss_function_spline_scaling, params_opt[4*N:6*N], method='Nelder-Mead', args=(params_opt[6*N:], params_opt[:4*N])).x
        params_opt[4*N:6*N] = params_spline_scaling_new_opt
    gaus_params_opt = params_opt[:4*N].reshape((N,-1))
    bh_opt = params_opt[4*N:5*N]
    bs_opt = params_opt[5*N:6*N]
    min_spline_sigma = np.min(bs_opt)
    tck[2] = params_opt[6*N:]
    gaus_amplitude_fit = curve_fit(dual_exponential, distances, gaus_params_opt[:,-1])[0]
    gaus_sigma_fit = curve_fit(sqrt_fit, distances, gaus_params_opt[:,-2])[0]
    spline_amplitude_fit = curve_fit(dual_exponential, distances, bh_opt)[0]
    spline_sigma_fit = curve_fit(sqrt_fit, distances, bs_opt)[0]
    # For kernel normalization
    L = dblquad(lambda x, y: bisplev(x, y, tck), x_data.min()/min_spline_sigma, x_data.max()/min_spline_sigma, y_data.min()/min_spline_sigma, y_data.max()/min_spline_sigma)[0]
    # Returned kernel
    return_function = lambda x, y, d, normalize=True, gaus_only=False, spline_only=False: kernel(x, y, d, dr, gaus_sigma_fit, gaus_amplitude_fit, spline_sigma_fit, spline_amplitude_fit, min_spline_sigma, tck, x_data, y_data, L, gaus_only=gaus_only, spline_only=spline_only, normalize=normalize)
    return return_function, params_opt