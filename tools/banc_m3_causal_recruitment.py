"""Causal sampled motor and muscle response for offline BANC recruitment fits."""

import math

import torch


def linear_response(forcing, dt, tau):
    """Zero-state exact first-order response to piecewise-linear forcing."""
    count = forcing.shape[0]
    tau = torch.as_tensor(tau, dtype=forcing.dtype, device=forcing.device)
    beta = torch.exp(-dt / tau)
    alpha = 1 - beta
    c = 1 - alpha / (dt / tau)
    impulse = torch.empty_like(forcing)
    impulse[0] = 0
    impulse[1:] = (alpha - c) * forcing[:-1] + c * forcing[1:]
    length = 1 << (2 * count - 2).bit_length()
    powers = torch.arange(count, device=forcing.device, dtype=forcing.dtype)
    kernel = beta.unsqueeze(0) ** powers[:, None]
    return torch.fft.irfft(torch.fft.rfft(impulse, n=length, dim=0)
        * torch.fft.rfft(kernel, n=length, dim=0), n=length, dim=0)[:count]


def endpoint_response(forcing, dt, tau):
    """Zero-state exact first-order response to endpoint zero-order-held forcing."""
    beta = math.exp(-dt / tau)
    impulse = torch.cat((torch.zeros_like(forcing[:1]), (1 - beta) * forcing[1:]))
    count = len(forcing)
    length = 1 << (2 * count - 2).bit_length()
    kernel = beta ** torch.arange(count, device=forcing.device, dtype=forcing.dtype)
    kernel = kernel.reshape((count,) + (1,) * (forcing.ndim - 1))
    return torch.fft.irfft(torch.fft.rfft(impulse, n=length, dim=0)
        * torch.fft.rfft(kernel, n=length, dim=0), n=length, dim=0)[:count]
