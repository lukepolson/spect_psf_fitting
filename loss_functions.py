import numpy as np
from scipy.ndimage import rotate

def rotation_loss(fit, dtheta=60):
    N = int(360/dtheta)
    rotations = np.array([rotate(fit, dtheta*i, reshape=False, cval=-1000, axes=(1,2)) for i in range(1,N)])
    fit = np.repeat(fit[np.newaxis], N-1, axis=0)
    return np.sum((fit[rotations>-999]-rotations[rotations>-999])**2)

def negativity_loss(fit):
    fit_negatives = fit.copy()
    fit_negatives[fit_negatives>0] = 0
    return np.sum(fit_negatives**2)