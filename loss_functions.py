import numpy as np
from scipy.ndimage import rotate

def loss_part1(params):
    return np.sum((gaus_part1(rv, params) - projectionss_data)**2)