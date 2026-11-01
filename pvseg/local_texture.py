"""Nine fixed local descriptors from uint8 RGB; no labels enter inference."""
import numpy as np

FEATURE_NAMES = (
    'fft_high_energy_ratio', 'fft_peak_energy_ratio', 'mean_red', 'mean_green',
    'mean_blue', 'mean_luma', 'std_luma', 'mean_gradient_magnitude',
    'coherent_gradient_energy',
)


def texture_features(rgb):
    """16x16 windows -> nine 56x56 maps, standardized within each image.

    Fourier descriptors use mean-centered, non-windowed 16x16 DFTs. The last
    descriptor is the squared mean gradient magnitude, not orientation
    anisotropy. These are basic descriptors, not a claim of a novel transform.
    """
    rgb = np.asarray(rgb)
    if rgb.shape != (896, 896, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected original uint8 RGB, shape 896,896,3')
    rgb = rgb.astype(np.float32) / 255.0
    y = .2126 * rgb[..., 0] + .7152 * rgb[..., 1] + .0722 * rgb[..., 2]
    patches = y.reshape(56, 16, 56, 16)
    rgb_mean = rgb.reshape(56, 16, 56, 16, 3).mean(axis=(1, 3)).transpose(2, 0, 1)
    local_mean = patches.mean(axis=(1, 3))
    local_std = patches.std(axis=(1, 3))
    gx = np.diff(y, axis=1, prepend=y[:, :1])
    gy = np.diff(y, axis=0, prepend=y[:1, :])
    grad = np.sqrt(gx * gx + gy * gy)
    grad_grid = grad.reshape(56, 16, 56, 16).mean(axis=(1, 3))
    gx_grid = gx.reshape(56, 16, 56, 16).mean(axis=(1, 3))
    gy_grid = gy.reshape(56, 16, 56, 16).mean(axis=(1, 3))
    coherent_energy = np.sqrt((gx_grid * gx_grid - gy_grid * gy_grid) ** 2
                              + (2 * gx_grid * gy_grid) ** 2)
    centered = patches - local_mean[:, None, :, None]
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered, axes=(1, 3)), axes=(1, 3))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(16))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(16))[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    total = power.sum(axis=(1, 3)) + 1e-6
    high_ratio = (power * (radius >= .25)[None, :, None, :]).sum(axis=(1, 3)) / total
    power[:, 8, :, 8] = 0
    peak_ratio = power.max(axis=(1, 3)) / total
    features = np.concatenate((high_ratio[None], peak_ratio[None], rgb_mean,
                               local_mean[None], local_std[None], grad_grid[None],
                               coherent_energy[None]), axis=0)
    features -= features.reshape(features.shape[0], -1).mean(axis=1)[:, None, None]
    features /= features.reshape(features.shape[0], -1).std(axis=1)[:, None, None] + 1e-5
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite local descriptors')
    return features.astype(np.float32)
