import numpy as np
from scipy.ndimage import convolve


def get_kernel(k):
    if k == 1:
        kernel = np.array([1/16, 1/4, 3/8, 1/4, 1/16])
    else:
        zeros = 2 ** (k - 1) - 1
        kernel = np.array((
                [1/16] + [0, ] * zeros +
                [1/4] + [0, ] * zeros +
                [3/8] + [0, ] * zeros +
                [1/4] + [0, ] * zeros +
                [1/16]
        ))
    return kernel


def convolve_3d(data, kernel):
    result = convolve(
        convolve(
            convolve(
                data,
                kernel[:, np.newaxis, np.newaxis],
                mode='mirror'
            ),
            kernel[np.newaxis, :, np.newaxis],
            mode='mirror'
        ),
        kernel[np.newaxis, np.newaxis, :],
        mode='mirror'
    )
    return result


def decompose(data, scales):
    image = data
    wavelets = np.empty((scales + 1, ) + data.shape, dtype=data.dtype)
    for k in range(1, scales + 1):
        new_image = convolve_3d(
            image,
            get_kernel(k)
        )
        wavelet = image - new_image
        wavelets[k - 1] = np.where(wavelet > 3 * wavelet.std(), wavelet, 0)
        image = new_image
    wavelets[-1] = image  # if any part is left store it at the last position
    return wavelets
