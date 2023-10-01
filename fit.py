import numpy as np
import torch
from torch.nn.functional import interpolate
from kornia.geometry.transform import rotate
import torch.nn.functional as F
from scipy.optimize import minimize

def fit(projectionss_data, idx=0):
    
    # Part 1: Get initial Gaussian parameters
    Nx = projectionss_data.shape[1]
    xv, yv = np.meshgrid(*2*[np.arange(-Nx/2+0.5, Nx/2+0.5, 1)])
    rv = np.sqrt(xv**2+yv**2)[np.newaxis]
    
    def gaus(x,params):
        params = params.reshape(-1,2)[:,:,np.newaxis,np.newaxis]
        return params[:,0]*np.exp(-x**2/(2*params[:,1]**2))
    
    def loss(params):
        return np.sum((gaus(rv, params) - projectionss_data)**2)
    
    gaus_params_init = np.ones(20)
    gaus_params_init[::2] = np.max(projectionss_data, axis=(1,2))
    gaus_params_opt = minimize(loss, gaus_params_init).x
    gaus_params_opt = gaus_params_opt.reshape(-1,2) 
    
    # Part 2: Get parameters for projections at each radial distance
    def gaus(x,params):
        return params[0]*torch.exp(-x**2/(2*params[1]**2))
    
    Nx = projectionss_data.shape[1]
    projectionss_data = torch.tensor(projectionss_data).unsqueeze(1)
    xv, yv = torch.meshgrid(*2*[torch.arange(-Nx/2+0.5, Nx/2+0.5, 1)])
    rv = torch.sqrt(xv**2+yv**2)
    g_params = torch.tensor(gaus_params_opt, requires_grad=True)
    g_params.retain_grad()
    w = torch.ones(1,1,Nx,requires_grad=True)*0.002
    w.retain_grad()
    iso = torch.ones(1,1,Nx,requires_grad=True)*0.0002
    iso.retain_grad()
    scale_factors = np.ones(projectionss_data.shape[0])
    amplitudes = np.ones(projectionss_data.shape[0])
    SA = np.vstack([scale_factors,amplitudes])
    
    def fit_func(w, iso, g_params, SA, idx):
        scale_factors = SA.reshape(2,-1)[0]
        amplitudes = SA.reshape(2,-1)[1]
        w_s = amplitudes[idx] * interpolate(w,scale_factor=scale_factors[idx], mode='linear')
        iso_s = amplitudes[idx] * interpolate(iso,scale_factor=scale_factors[idx], mode='linear')
        x = gaus(rv, g_params[idx]).unsqueeze(1)
        y = F.conv1d(x,w_s,padding='same')
        tot = torch.clone(x)
        #TODO: Update for non-hexagonal configs
        for i in range(0,3):
            angle = torch.tensor(60*i).to(torch.float)
            tot += rotate(y.unsqueeze(0).swapaxes(1,2), angle, mode='bilinear').swapaxes(1,2).squeeze(0)
        bkg = F.conv1d(x,iso_s,padding='same')
        bkg = F.conv1d(bkg.swapaxes(2,0),iso_s,padding='same').swapaxes(2,0)
        tot+=bkg
        tot = tot.swapaxes(0,1)
        return tot
    
    def train_w(w, iso, g_params, SA, n_iters, lr=1e-3, idx=None):
        for i in range(n_iters):
            if w.grad is not None:
                w.grad.zero_()
            if iso.grad is not None:
                iso.grad.zero_()
            if g_params.grad is not None:
                g_params.grad.zero_()
            error = 0
            for i in range(projectionss_data.shape[0]):
                if idx is not None:
                    if i!=idx:
                        continue
                pred = fit_func(w, iso, g_params, SA, i)
                error += torch.sum((pred-projectionss_data[i])**2) + 100*torch.sum(w[w<0]**2) + 10*torch.sum(torch.diff(SA[1,i]*w)**2) + 10*torch.sum(torch.diff(SA[1,i]*iso)**2)
            error.backward()
            w = (w.data - lr * w.grad).requires_grad_(True)
            iso = (iso.data - lr * iso.grad).requires_grad_(True)
            g_params = (g_params.data - lr * g_params.grad).requires_grad_(True)
        return w, iso, g_params
    
    w, iso, g_params = train_w(w, iso, g_params, SA, n_iters=1000, idx=idx)
    
    # Adjust other kernels
    def loss(SA):
        err = 0
        for i in range(10):
            pred = fit_func(w, iso, g_params, SA, i)
            err += np.sum((pred.detach().cpu().numpy() - projectionss_data[i].detach().cpu().numpy())**2)
        return err
    
    #TODO: Adjust parameters such as number of loops, learning rate, n_iters
    for n in range(20):
        lr = 2e-4 / (n/3+1)
        SA = minimize(loss, SA.ravel(), method='COBYLA').x
        SA = SA.reshape(2,-1)
        w, iso, g_params = train_w(w, iso, g_params, SA, n_iters=200, lr=lr)
        

    return w, iso, g_params, SA, fit_func