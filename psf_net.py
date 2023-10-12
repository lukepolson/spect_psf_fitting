import numpy as np
import torch
from kornia.geometry.transform import rotate
import pytomography
from pytomography.utils import pad_object_z, unpad_object_z
import torch.nn as nn
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from fft_conv_pytorch import FFTConv1d, FFTConv2d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dual_exponential(d, b1, b2, b3, b4):
    return b1*np.exp(-b2*d)+b3*np.exp(-b4*d)

def sqrt_fit(d, b1, b2):
    return b1*d+b2

# Used for timing comparison in paper: not for implementation of PSFNet
class JaggedConv2D(nn.Module):
    def __init__(self, psf_net, fft=False):
        super(JaggedConv2D, self).__init__()
        self.fft = fft
        self.psf_net = psf_net
            
    def configure(self, distances, dr):
        self.psf_net.configure(distances, dr)
        kernel_sizes = self.psf_net.kernel_sizes
        kernel_size_max = self.psf_net.kernel_size_max
        x = torch.zeros((1,len(distances),kernel_size_max, kernel_size_max)).to(pytomography.device)
        x[:,:,kernel_size_max//2,kernel_size_max//2] = 1
        kernel = self.psf_net(x, padding=False)
        
        # Create a list of convolution layers with different kernel sizes
        self.conv_layers = nn.ModuleList()
        for i, kernel_size in enumerate(kernel_sizes):
            if self.fft:
                layer = FFTConv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding='same', padding_mode='constant', bias=0)
            else:
                layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding='same', padding_mode='zeros', bias=0)
            start_idx = (kernel.shape[2] - kernel_size) // 2
            layer.weight.data = kernel[:,i,start_idx:start_idx+kernel_size,start_idx:start_idx+kernel_size].unsqueeze(0)
            self.conv_layers.append(layer)
            
    @torch.no_grad()
    def forward(self, x):
        input_channels = x.split(1, dim=1)
        conv_results = [conv(input_channel) for conv, input_channel in zip(self.conv_layers, input_channels)]
        output = torch.cat(conv_results, dim=1)
        return output

class PSFNet(nn.Module):
    def __init__(self, gaus_amplitude_fit, gaus_sigma_fit, mini_gaus_amplitude_fit, mini_gaus_sigma_fit, bkg_amplitude_fit, bkg_sigma_fit, w_spline_f, iso_spline_f, dr0, kernel_size0):
        super(PSFNet, self).__init__()
        self.gaus_amplitude_fit = gaus_amplitude_fit
        self.gaus_sigma_fit = gaus_sigma_fit
        self.mini_gaus_amplitude_fit = mini_gaus_amplitude_fit
        self.mini_gaus_sigma_fit = mini_gaus_sigma_fit
        self.bkg_amplitude_fit = bkg_amplitude_fit
        self.bkg_sigma_fit = bkg_sigma_fit
        self.w_spline_f = w_spline_f
        self.iso_spline_f = iso_spline_f
        self.dr0 = dr0
        self.kernel_size0 = kernel_size0
        self.gaus_only = False
    
    def set_gaus_only(self, b):
        self.gaus_only = b
        
    def configure(self, distances, dr, fft=False):
        self.fft = fft
        self.distances = distances
        self.dr = dr
        self.gaus_amplitudes = dual_exponential(self.distances, *self.gaus_amplitude_fit)
        self.gaus_sigmas = sqrt_fit(self.distances, *self.gaus_sigma_fit)
        self.mini_gaus_amplitudes = dual_exponential(self.distances, *self.mini_gaus_amplitude_fit)
        self.mini_gaus_sigmas = sqrt_fit(self.distances, *self.mini_gaus_sigma_fit)
        self.bkg_amplitudes = dual_exponential(self.distances, *self.bkg_amplitude_fit)
        self.bkg_sigmas = sqrt_fit(self.distances, *self.bkg_sigma_fit)
        self.kernel_sizes = np.ceil(self.bkg_sigmas * self.dr0/dr * self.kernel_size0).astype(int)
        self.kernel_sizes[self.kernel_sizes%2==0] +=1
        self.kernel_size_max = self.kernel_sizes.max()
        # Get normalization factor
        Iw = self.w_spline_f.integrate(min(self.w_spline_f.x), max(self.w_spline_f.x))
        Iiso = self.iso_spline_f.integrate(min(self.iso_spline_f.x), max(self.iso_spline_f.x))
        Nw = 3*self.bkg_amplitudes* Iw * self.bkg_sigmas 
        Niso = self.bkg_amplitudes *Iiso * self.bkg_sigmas
        Ng2 = (self.gaus_amplitudes * 2 * np.pi * self.gaus_sigmas**2) * (self.dr0 / self.dr)**2
        Ng2_mini = (self.mini_gaus_amplitudes * 2 * np.pi * self.mini_gaus_sigmas**2) * (self.dr0 / self.dr)**2
        normalization_factor = (Nw+Niso**2+1)*(Ng2+Ng2_mini)
        self.normalization_factor = torch.tensor(normalization_factor).to(pytomography.device)
        # Now get layers
        self.layer_g = self.get_gaus_layer(self.gaus_amplitudes, self.gaus_sigmas)
        self.layer_mini_g = self.get_gaus_layer(self.mini_gaus_amplitudes, self.mini_gaus_sigmas)
        self.layer_w = self.get_bkg_layers(self.w_spline_f)
        self.layer_iso = self.get_bkg_layers(self.iso_spline_f)
        
        
    def get_gaus_layer(self, A, sigma, max_sigmas=3):
        N = len(self.distances)
        kernel_size_gaus_max = 2*(np.ceil(max_sigmas*max(sigma))) + 1
        x = torch.arange(-int(kernel_size_gaus_max//2), int(kernel_size_gaus_max//2)+1).to(pytomography.device).unsqueeze(0).unsqueeze(0).repeat((N,1,1)) * self.dr / self.dr0
        gaus_amplitudes = torch.tensor(A).to(pytomography.device).to(pytomography.dtype).reshape((N,1,1))
        gaus_sigmas = torch.tensor(sigma).to(pytomography.device).to(pytomography.dtype).reshape((N,1,1))
        kernel = torch.sqrt(gaus_amplitudes) * torch.exp(-x**2 / (2*gaus_sigmas**2 + pytomography.delta))
        if self.fft:
            layer = FFTConv1d(N, N, self.kernel_size_max, groups=N, padding='same', padding_mode='constant', bias=0).to(pytomography.device)
        else:
            layer = nn.Conv1d(N, N, self.kernel_size_max, groups=N, padding='same', padding_mode='zeros', bias=0, device=pytomography.device)
        layer.weight.data = kernel.to(pytomography.dtype)
        return layer

    def get_bkg_layers(self, spline_f):
        N = len(self.distances)
        bkg_amplitudes = torch.tensor(self.bkg_amplitudes).to(pytomography.device).to(pytomography.dtype).reshape((N,1,1))
        bkg_sigmas = torch.tensor(self.bkg_sigmas).to(pytomography.device).to(pytomography.dtype).reshape((N,1,1))
        x = torch.arange(-int(self.kernel_size_max//2), int(self.kernel_size_max//2)+1).to(pytomography.device).unsqueeze(0).unsqueeze(0).repeat((N,1,1))  * self.dr / self.dr0 / bkg_sigmas
        components = spline_f(x.cpu().numpy())
        components[np.isnan(components)] = 0
        kernel = bkg_amplitudes * self.dr/self.dr0 * torch.tensor(components).to(pytomography.device)
        if self.fft:
            layer = FFTConv1d(N, N, self.kernel_size_max, groups=N, padding='same', padding_mode='constant', bias=0).to(pytomography.device)
        else:
            layer = nn.Conv1d(N, N, self.kernel_size_max, groups=N, padding='same', padding_mode='zeros', bias=0, device=pytomography.device)
        layer.weight.data = kernel.to(pytomography.dtype)
        return layer

    @torch.no_grad()
    def forward(self, input, norm=True, padding=True):
        output = input[0] #xyz, only works with batchsize=1
        output1 = self.layer_g(output.permute(2,0,1)).permute(1,2,0)
        output1 = self.layer_g(output1.permute(1,0,2)).permute(1,0,2)
        if self.gaus_only:
            output1 /= self.normalization_factor.unsqueeze(1).unsqueeze(1)
            return output1.unsqueeze(0)
        output2 = self.layer_mini_g(output.permute(2,0,1)).permute(1,2,0)
        output2 = self.layer_mini_g(output2.permute(1,0,2)).permute(1,0,2)
        output = output1+output2
        iso = self.layer_iso(output.permute(2,0,1)).permute(1,2,0)
        iso = self.layer_iso(iso.permute(1,0,2)).permute(1,0,2)
        tails = 0
        # Make object square before rotating
        #TODO: What if z is bigger than y/x??
        pad_size = int((output.shape[0] - output.shape[2])/2)
        for i in range(3):
            angle = torch.tensor(60*i).to(torch.float).to(device)
            temp = output.unsqueeze(0)
            if padding*(pad_size>0):
                temp = pad_object_z(temp, pad_size)
            temp = rotate(temp, angle, mode='bilinear').squeeze()
            tails += rotate(self.layer_w(temp.permute(1,0,2)).permute(1,0,2).unsqueeze(0), -angle, mode='bilinear')
        if padding*(pad_size>0):
            tails = unpad_object_z(tails, pad_size)
        tails = tails.squeeze()
        output = (output  + tails + iso)
        if norm:
            output /= self.normalization_factor.unsqueeze(1).unsqueeze(1)
        return output.unsqueeze(0)
    
def get_psf_net(distances, dr0, projectionss_data, w, iso, g_params, mini_g_params, SA, kernel_size0):
    bkg_amplitude = SA[1]
    bkg_sigma = SA[0]
    gaus_amplitude = g_params[:,0].detach().cpu().numpy()
    gaus_sigma = g_params[:,1].detach().cpu().numpy()
    mini_gaus_amplitude = mini_g_params[:,0].detach().cpu().numpy()
    mini_gaus_sigma = mini_g_params[:,1].detach().cpu().numpy()
    
    gaus_amplitude_fit = curve_fit(dual_exponential, distances, gaus_amplitude)[0]
    gaus_sigma_fit = curve_fit(sqrt_fit, distances, gaus_sigma)[0]
    mini_gaus_amplitude_fit = curve_fit(dual_exponential, distances, mini_gaus_amplitude)[0]
    mini_gaus_sigma_fit = curve_fit(sqrt_fit, distances, mini_gaus_sigma)[0]
    bkg_amplitude_fit = curve_fit(dual_exponential, distances, bkg_amplitude)[0]
    bkg_sigma_fit = curve_fit(sqrt_fit, distances, bkg_sigma)[0]
    
    Nx = projectionss_data.shape[1]
    x = np.arange(-Nx/2+0.5, Nx/2+0.5, 1)
    w_spline_f = CubicSpline(x, w[0,0].detach().cpu().numpy(), extrapolate=False)
    iso_spline_f = CubicSpline(x, iso[0,0].detach().cpu().numpy(), extrapolate=False)
    return PSFNet(gaus_amplitude_fit, gaus_sigma_fit, mini_gaus_amplitude_fit, mini_gaus_sigma_fit, bkg_amplitude_fit, bkg_sigma_fit, w_spline_f, iso_spline_f, dr0, kernel_size0)