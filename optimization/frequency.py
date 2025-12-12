import torch


def ifft(data, ndim, norm=True):
    """Perform inverse FFT on shifted frequency data."""
    dims = tuple(range(-ndim, 0))
    f = torch.fft.ifftshift(data, dim=dims)
    norm_mode = "ortho" if norm else "backward"

    if ndim < 4:
        grid = torch.fft.ifftn(f, dim=dims, norm=norm_mode)
    else:
        # 4D: combine 3D spatial + 1D temporal FFT
        grid = torch.fft.ifftn(f, dim=(-3, -2, -1), norm=norm_mode)
        grid = torch.fft.ifftn(grid, dim=(1,), norm=norm_mode)

    return grid.real
