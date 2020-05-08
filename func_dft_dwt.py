import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def fft_wavenumbers(x, y, shape_dat, shape_pdat):
    """
    Compute the wavenumbers.

    Parameters
    ----------
    x : 1D array
        Coordinates along x direction.
    y : 1D array
        Coordinates along y direction.
    shape_dat : tuple
        Shape of the input data.
    shape_pdat : tuple
        Shape of the pad.

    Returns
    -------
    u : array
        Wavenumber.
    v : TYPE
        Wavenumber.
    """

    dx = (np.amax(x) - np.amin(x))/(shape_dat[0] - 1)
    dy = (np.amax(y) - np.amin(y))/(shape_dat[1] - 1)
    fx = 2*np.pi*np.fft.fftfreq(shape_pdat[0], dx)
    fy = 2*np.pi*np.fft.fftfreq(shape_pdat[1], dy)
    v,u=np.meshgrid(fy, fx)   
    
    return (u,v)

def fft_pad_data(data, mode='edge'):
    """
    Perform the 2D discrete Fourier transform and extend the data with padding.

    Parameters
    ----------
    data : 2D array
        Input data.
    mode : TYPE, optional
        The type of the pad, available on numpy.pad. The default is 'edge'.

    Returns
    -------
    fpdat : 2D array
        The padded data.
    mask : boolean
        The mask to perform the unppading.
    """

    n_points=int(2**(np.ceil(np.log(np.max(data.shape))/np.log(2))))
    nx, ny = data.shape    
    padx = int((n_points - nx)/2)
    pady = int((n_points - ny)/2)
    
    padded_data = np.pad(data, ((padx, padx), (pady, pady)),mode)    
    
    mask = np.zeros_like(padded_data, dtype=bool)
    mask[padx:padx+data.shape[0], pady:pady+data.shape[1]] = True 
    fpdat = np.fft.fft2(padded_data)

    return (fpdat,mask)

def ifft_unpad_data(data_p, mask, shape_dat):
    '''
    Unpad the extended data to fit the original data shape.

    Parameters
    ----------
    data_p : 2D array
        Padded data.
    mask : boolean
        The mask that will be used to unpad the data.
    shape_dat : tuple
        Shape of the original data.

    Returns
    -------
    data : array
        Unpadded data.

    '''
    
    ifft_data = np.real(np.fft.ifft2(data_p))
    data = ifft_data[mask]
    return np.reshape(data, shape_dat)
	
def butter2d_lp(shape, f, n): 
    """
    Designs a lowpass 2D Butterworth filter.
    Modified from Peirce JW (2009) Generating stimuli for neuroscience using
    PsychoPy. Front. Neuroinform. 2:10.
    doi:10.3389/neuro.11.010.2008.

    Parameters
    ----------
    shape : tuple
        Size of the filter.
    f : float
        Relative cutoff frequency of the filter.
    n : int
        Order of the filter, the higher n is the sharper the transition is.

    Returns
    -------
    filt : 2D array
        Filter kernel centered.

    """
    
    rows, cols = shape 
    x = np.linspace(-0.5, 0.5, cols)
    y = np.linspace(-0.5, 0.5, rows)
    radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis]) 
    filt = 1 / (1.0 + (radius / f)**(2*n))
    
    return (filt)	

def plot_wav(decomp):
    """
    Plot the data in DWT domain

    Parameters
    ----------
    data : list
        Data in wavelet domain.

    Returns
    -------
    None.

    """
    
    plt.figure(figsize=(10,10))
    gs = GridSpec(4, 4)
    
    ax = plt.subplot(gs[0, 0])
    plt.imshow(decomp[0])
    plt.xticks([])
    plt.yticks([])
    
    ax = plt.subplot(gs[1,0])
    plt.imshow(decomp[1][0])
    plt.xticks([])
    plt.yticks([])
    
    ax = plt.subplot(gs[0, 1])
    plt.imshow(decomp[1][1])
    plt.xticks([])
    plt.yticks([])
    
    ax = plt.subplot(gs[1, 1])
    plt.imshow(decomp[1][2])
    plt.xticks([])
    plt.yticks([])
    
    ax = plt.subplot(gs[2:,:2])
    plt.imshow(decomp[2][0])
    plt.xticks([])
    plt.yticks([])
    
    ax = plt.subplot(gs[:2,2:])
    plt.imshow(decomp[2][1])
    plt.xticks([])
    plt.yticks([])
    
    ax = plt.subplot(gs[2:,2:])
    plt.imshow(decomp[2][2])
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    
    return