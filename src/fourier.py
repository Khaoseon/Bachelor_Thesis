from __future__ import annotations

import math

import torch


def conjugate_frequency(freq_i: int, freq_j: int, size: int) -> tuple[int, int]:
    """
    Return the Hermitian-symmetric frequency of (freq_i, freq_j).

    For a real spatial perturbation, Fourier coefficients must satisfy
    X[i, j] = conj(X[-i mod N, -j mod N]).
    """
    return (-freq_i) % size, (-freq_j) % size


def fourier_basis_2d(
    size: int,
    freq_i: int,
    freq_j: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    normalized: bool = True,
) -> torch.Tensor:
    """
    Build the complex 2D Fourier basis indexed by (freq_i, freq_j).

    The spatial tensor has shape (size, size). Axis 0 corresponds to rows and
    axis 1 to columns. With normalized=True, the Frobenius norm is 1, matching
    the outer product of two normalized 1D Fourier vectors.
    """
    if size <= 0:
        raise ValueError("size must be positive.")

    freq_i = freq_i % size
    freq_j = freq_j % size
    rows = torch.arange(size, device=device, dtype=dtype).view(size, 1)
    cols = torch.arange(size, device=device, dtype=dtype).view(1, size)
    phase = 2.0 * math.pi * ((freq_i * rows / size) + (freq_j * cols / size))
    basis = torch.exp(1j * phase)
    if normalized:
        basis = basis / size
    return basis


def real_fourier_pattern_2d(
    size: int,
    freq_i: int,
    freq_j: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build the real spatial direction used by Single Fourier Attack.

    This follows Algorithm 1 of Tsuzuku and Sato in direction form:
    (1 + i) F_i x F_j + (1 - i) F_{-i} x F_{-j}.
    The conjugate pair makes the resulting perturbation real-valued.
    """
    basis = fourier_basis_2d(
        size,
        freq_i,
        freq_j,
        device=device,
        dtype=dtype,
        normalized=True,
    )
    conj_i, conj_j = conjugate_frequency(freq_i, freq_j, size)
    conjugate_basis = fourier_basis_2d(
        size,
        conj_i,
        conj_j,
        device=device,
        dtype=dtype,
        normalized=True,
    )
    pattern = ((1.0 + 1.0j) * basis + (1.0 - 1.0j) * conjugate_basis).real
    return pattern.to(dtype=dtype)


def normalize_linf(pattern: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Scale a perturbation direction so that ||pattern||_inf == epsilon.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")
    max_abs = pattern.abs().amax()
    if float(max_abs) == 0.0:
        raise ValueError("Cannot normalize a zero perturbation.")
    return pattern * (epsilon / max_abs)


def single_fourier_noise(
    size: int,
    freq_i: int,
    freq_j: int,
    epsilon: float,
    *,
    channels: int = 3,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return an RGB Single Fourier Attack perturbation.

    The same real Fourier pattern is added to each channel, as in Algorithm 1
    where the update is applied for each image channel. The returned tensor has
    shape (channels, size, size) and satisfies ||noise||_inf = epsilon when
    epsilon > 0.
    """
    if channels <= 0:
        raise ValueError("channels must be positive.")

    pattern = real_fourier_pattern_2d(
        size,
        freq_i,
        freq_j,
        device=device,
        dtype=dtype,
    )
    noise_2d = normalize_linf(pattern, epsilon)
    return noise_2d.unsqueeze(0).expand(channels, -1, -1).contiguous()


def apply_perturbation(images: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    Apply a Fourier perturbation in image space and clip to [0, 1].

    images must be unnormalized image tensors in [0, 1], with shape (C, H, W)
    or (B, C, H, W). Normalization for the model must happen after this step.
    """
    if images.ndim == 3:
        return torch.clamp(images + noise, 0.0, 1.0)
    if images.ndim == 4:
        return torch.clamp(images + noise.unsqueeze(0), 0.0, 1.0)
    raise ValueError("images must have shape (C, H, W) or (B, C, H, W).")


def fool_ratio(clean_predictions: torch.Tensor, perturbed_predictions: torch.Tensor) -> float:
    """
    Compute the fool ratio used in the paper.

    The metric is the fraction of images whose predicted class changes:
    pred_clean != pred_perturbed. Ground-truth labels are not used.
    """
    if clean_predictions.shape != perturbed_predictions.shape:
        raise ValueError("Prediction tensors must have the same shape.")
    if clean_predictions.numel() == 0:
        raise ValueError("Cannot compute fool ratio on an empty tensor.")
    changed = clean_predictions != perturbed_predictions
    return changed.float().mean().item()
