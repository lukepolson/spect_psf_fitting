import numpy as np
from copy import deepcopy
import numpy as np
from scipy.interpolate import bisplev
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from fit_functions import dual_exponential, sqrt_fit, gaus, initial_gaus_estimate
from loss_functions import negativity_loss
from load_data import get_projections_spacing_radius

# Unfortunately distances have to be provided seperately because not stored in STATIC SPECT scan .h00 files (single projection)
def get_psf_kernel(projection_paths):
    psfs_data = []
    distances = []
    for projection_path in projection_paths:
        projections, dr, distance = get_projections_spacing_radius(projection_path)
        psfs_data.append(projections)
        distances.append(distance)
    return fit(np.array(psfs_data), np.array(distances), dr)[0]
        
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
    tck[2] = params_opt[6*N:]
    gaus_amplitude_fit = curve_fit(dual_exponential, distances, gaus_params_opt[:,-1])[0]
    gaus_sigma_fit = curve_fit(sqrt_fit, distances, gaus_params_opt[:,-2])[0]
    spline_amplitude_fit = curve_fit(dual_exponential, distances, bh_opt)[0]
    spline_sigma_fit = curve_fit(sqrt_fit, distances, bs_opt)[0]
    def return_function(x, y, d, gaus_only=False, spline_only=False):
        x_new = x/dr[0] # in units of voxels corresponding to original fit
        y_new = y/dr[1] # in units of voxels corresponding to original fit
        x_grid, y_grid = np.meshgrid(x_new, y_new)
        gaus_comp = gaus(x_grid, y_grid, [0, 0, sqrt_fit(d, *gaus_sigma_fit), dual_exponential(d, *gaus_amplitude_fit)])
        spline_comp = dual_exponential(d, *spline_amplitude_fit)*bisplev(x_new/sqrt_fit(d, *spline_sigma_fit), y_new/sqrt_fit(d, *spline_sigma_fit), tck)
        spline_comp[(x_grid>x_data.max())+(x_grid<x_data.min())+(y_grid>y_data.max())+(y_grid<y_data.min())] = 0 # set outside fitted boundaries equal to zero
        if gaus_only:
            return gaus_comp
        elif spline_only:
            return spline_comp
        else:
            return gaus_comp + spline_comp
    return return_function, params_opt