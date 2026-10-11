#!/usr/bin/env python3
"""Build the four deterministic English-localized report figures."""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "report" / "figures"
CIFAR_INPUTS = {
    "CIFAR-10": (
        Path("cifar-10-batches-py/test_batch"),
        "40351d587109b95175f43aff81a1287e",
    ),
    "CIFAR-100": (
        Path("cifar-100-python/test"),
        "f0ef6b0ae62326f3e7ffdfab6717acfc",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the four approved English-localized report figures."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the extracted CIFAR-10 and CIFAR-100 archives.",
    )
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def build_fourier_basis(output_path: Path) -> None:
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
        }
    )

    patch_size = 8
    grid_size = 9
    canvas_real = np.full((grid_size * 9, grid_size * 9), np.nan)
    canvas_imag = np.full((grid_size * 9, grid_size * 9), np.nan)
    x, y = np.meshgrid(np.arange(patch_size), np.arange(patch_size))

    for row, u in enumerate(range(-4, 5)):
        for column, v in enumerate(range(-4, 5)):
            angle = 2 * np.pi * (u * y / patch_size + v * x / patch_size)
            row_start = row * 9
            column_start = column * 9
            canvas_real[
                row_start : row_start + patch_size,
                column_start : column_start + patch_size,
            ] = np.cos(angle)
            canvas_imag[
                row_start : row_start + patch_size,
                column_start : column_start + patch_size,
            ] = -np.sin(angle)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        facecolor="white",
        constrained_layout=True,
    )
    axes[0].set_facecolor("white")
    image = axes[0].imshow(canvas_real, cmap="gray", vmin=-1, vmax=1)
    axes[0].set_title("Centered Real Part", pad=15)
    axes[0].axis("off")
    axes[1].set_facecolor("white")
    axes[1].imshow(canvas_imag, cmap="gray", vmin=-1, vmax=1)
    axes[1].set_title("Centered Imaginary Part", pad=15)
    axes[1].axis("off")
    colorbar = fig.colorbar(
        image,
        ax=axes,
        orientation="vertical",
        shrink=0.8,
        fraction=0.05,
        pad=0.04,
    )
    colorbar.set_label("Amplitude", rotation=90, labelpad=15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)


def build_periodicity_and_gibbs(output_path: Path) -> None:
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    main_color = "#334155"
    accent_color = "#0d9488"
    light_color = "#94a3b8"
    text_color = "#1e293b"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.color": text_color,
            "axes.labelcolor": text_color,
            "axes.titlecolor": text_color,
            "axes.titlepad": 25,
        }
    )

    fig, axes = plt.subplot_mosaic(
        [["nonperiodic", "periodic"], ["gibbs", "gibbs"]],
        figsize=(14, 10),
        gridspec_kw={"hspace": 0.35},
    )

    def clean_centered_axes(axis: plt.Axes) -> None:
        axis.grid(False)
        axis.spines["left"].set_position("zero")
        axis.spines["bottom"].set_position("zero")
        axis.spines["left"].set_color(light_color)
        axis.spines["bottom"].set_color(light_color)
        axis.spines["right"].set_color("none")
        axis.spines["top"].set_color("none")
        axis.set_xticks([])
        axis.set_yticks([])

    x_nonperiodic = np.linspace(-10, 10, 4000)
    signal_nonperiodic = np.exp(-(x_nonperiodic**2) / 8) * np.cos(4 * x_nonperiodic)
    spacing = x_nonperiodic[1] - x_nonperiodic[0]
    frequencies = np.fft.fftshift(np.fft.fftfreq(len(x_nonperiodic), d=spacing))
    spectrum = np.fft.fftshift(np.abs(np.fft.fft(signal_nonperiodic)))
    spectrum /= spectrum.max()

    axis = axes["nonperiodic"]
    axis.plot(x_nonperiodic, signal_nonperiodic, linewidth=1.5, color=main_color)
    axis.set_title("Signal non périodique\n(support non borné)")
    clean_centered_axes(axis)
    inset = axis.inset_axes([0.60, 0.65, 0.38, 0.28])
    inset.plot(frequencies, spectrum, linewidth=1.5, color=main_color)
    inset.set_title("Spectre continu", fontsize=10, pad=10)
    inset.grid(True, linestyle="--", alpha=0.3, color=light_color)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color(light_color)
        spine.set_linewidth(0.5)

    x_periodic = np.linspace(-np.pi, np.pi, 1000)
    signal_periodic = (
        np.sin(x_periodic)
        + 0.6 * np.cos(2 * x_periodic)
        + 0.35 * np.sin(4 * x_periodic)
    )
    axis = axes["periodic"]
    axis.plot(x_periodic, signal_periodic, linewidth=1.5, color=main_color)
    axis.set_title("Signal périodique")
    clean_centered_axes(axis)
    inset = axis.inset_axes([0.60, 0.07, 0.38, 0.28])
    harmonics = np.array([1, 2, 4])
    amplitudes = np.array([1.0, 0.6, 0.35])
    markerline, stemlines, baseline = inset.stem(harmonics, amplitudes)
    plt.setp(markerline, markersize=5, color=main_color)
    plt.setp(stemlines, linewidth=1.5, color=main_color)
    plt.setp(baseline, visible=False)
    inset.set_title("Spectre discret", fontsize=10, pad=10)
    inset.grid(True, linestyle="--", alpha=0.3, color=light_color)
    inset.set_xticks(harmonics)
    inset.set_yticks([])
    inset.tick_params(axis="x", labelsize=8, colors=main_color)
    for spine in inset.spines.values():
        spine.set_color(light_color)
        spine.set_linewidth(0.5)

    x_gibbs = np.linspace(-np.pi, np.pi, 3000)
    step = np.where(x_gibbs >= 0, 1.0, 0.0)
    partial_sum = 0.5 * np.ones_like(x_gibbs)
    for n in range(1, 26):
        if n % 2 == 1:
            partial_sum += (2 / (np.pi * n)) * np.sin(n * x_gibbs)

    axis = axes["gibbs"]
    axis.plot(
        x_gibbs,
        step,
        linewidth=2,
        color=light_color,
        label="Signal périodisé",
        alpha=0.4,
    )
    axis.plot(
        x_gibbs,
        partial_sum,
        linewidth=1.5,
        color=accent_color,
        label="Somme partielle (N=25)",
    )
    axis.set_title("Support fini, bord et phénomène de Gibbs")
    axis.spines["left"].set_position("zero")
    axis.spines["bottom"].set_position("zero")
    axis.spines["left"].set_color(light_color)
    axis.spines["bottom"].set_color(light_color)
    axis.spines["right"].set_color("none")
    axis.spines["top"].set_color("none")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_ylim(-0.3, 1.3)
    axis.legend(frameon=False, loc="lower right", fontsize=10)
    axis.annotate(
        "Discontinuité / bord raccordé\n(Phénomène de Gibbs)",
        xy=(0, 1.0),
        xytext=(-2.6, 0.75),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1,
            "color": text_color,
            "connectionstyle": "arc3,rad=-0.2",
        },
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "white",
            "ec": light_color,
            "alpha": 0.8,
        },
    )

    plt.tight_layout()
    fig.canvas.draw()
    positions = [plot_axis.get_position().frozen() for plot_axis in fig.axes]
    fig.set_layout_engine(None)
    for plot_axis, position in zip(fig.axes, positions):
        plot_axis.set_position(position)
    axes["nonperiodic"].set_title("Non-periodic signal\n(unbounded support)")
    axes["nonperiodic"].child_axes[0].set_title(
        "Continuous spectrum", fontsize=10, pad=10
    )
    axes["periodic"].set_title("Periodic signal")
    axes["periodic"].child_axes[0].set_title(
        "Discrete spectrum", fontsize=10, pad=10
    )
    axes["gibbs"].set_title("Finite support, boundary, and Gibbs phenomenon")
    legend = axes["gibbs"].get_legend()
    legend.get_texts()[0].set_text("Periodized signal")
    legend.get_texts()[1].set_text("Partial sum (N=25)")
    axes["gibbs"].texts[0].set_text(
        "Joined discontinuity / boundary\n(Gibbs phenomenon)"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def load_cifar_images(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        payload: dict[str, Any] = pickle.load(handle, encoding="latin1")
    data = np.asarray(payload["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    if data.shape != (10000, 3, 32, 32):
        raise ValueError(f"Unexpected CIFAR test-set shape in {path}: {data.shape}")
    return torch.from_numpy(data.copy()).to(dtype=torch.float32).div_(255.0)


def mean_clean_spectrum(images: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    spectrum_sum: torch.Tensor | None = None
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size]
        magnitude = torch.abs(torch.fft.fft2(batch, dim=(-2, -1)))
        magnitude = torch.fft.fftshift(magnitude, dim=(-2, -1))
        spectrum_per_image = torch.log1p(magnitude).mean(dim=1)
        batch_sum = spectrum_per_image.sum(dim=0)
        spectrum_sum = batch_sum if spectrum_sum is None else spectrum_sum + batch_sum
    if spectrum_sum is None:
        raise ValueError("The CIFAR input is empty.")
    return (spectrum_sum / images.shape[0]).cpu().numpy()


def build_mean_clean_spectrum(data_dir: Path, output_path: Path) -> None:
    plt.rcdefaults()
    inputs = {
        name: (data_dir / relative_path, expected_md5)
        for name, (relative_path, expected_md5) in CIFAR_INPUTS.items()
    }
    for path, expected_md5 in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_md5 = hashlib.md5(path.read_bytes()).hexdigest()
        if actual_md5 != expected_md5:
            raise ValueError(
                f"Unexpected checksum for {path}: {actual_md5} != {expected_md5}"
            )

    results = {
        name: mean_clean_spectrum(load_cifar_images(path))
        for name, (path, _expected_md5) in inputs.items()
    }
    global_min = min(float(spectrum.min()) for spectrum in results.values())
    global_max = max(float(spectrum.max()) for spectrum in results.values())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    image = None
    for axis, (name, spectrum) in zip(axes, results.items()):
        image = axis.imshow(
            spectrum,
            origin="lower",
            cmap="magma",
            vmin=global_min,
            vmax=global_max,
        )
        axis.set_title(f"{name}\n(10000 images)")
        axis.set_xlabel("frequency $v$")
        axis.set_ylabel("frequency $u$")
        axis.set_xticks([0, 16, 31])
        axis.set_yticks([0, 16, 31])
    if image is None:
        raise RuntimeError("No spectrum was plotted.")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.04)
    colorbar.set_label(r"$\log(1 + |\mathcal{F}(x)|)$")
    fig.suptitle("Mean Fourier spectrum of clean images (CIFAR)", fontsize=16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_spatial_sfa_pattern(output_path: Path) -> None:
    """Render the validated CIFAR-10 SFA pattern at frequency (16, 24)."""
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    size = 32
    frequency = (16, 24)
    epsilon = 10 / 255
    rows = torch.arange(size, dtype=torch.float32).view(size, 1)
    columns = torch.arange(size, dtype=torch.float32).view(1, size)

    def fourier_basis(index: tuple[int, int]) -> torch.Tensor:
        phase = 2 * torch.pi * (
            index[0] * rows / size + index[1] * columns / size
        )
        return torch.exp(1j * phase) / size

    basis = fourier_basis(frequency)
    conjugate_basis = fourier_basis(
        ((-frequency[0]) % size, (-frequency[1]) % size)
    )
    pattern = ((1 + 1j) * basis + (1 - 1j) * conjugate_basis).real
    pattern *= epsilon / pattern.abs().amax()
    pattern_array = pattern.numpy()
    display = (pattern_array - pattern_array.min()) / (
        pattern_array.max() - pattern_array.min()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.9, 3.9),
        constrained_layout=True,
    )
    image = axes[0].imshow(
        pattern_array,
        origin="lower",
        cmap="coolwarm",
        vmin=-epsilon,
        vmax=epsilon,
    )
    axes[0].set_title("Real perturbation\nbounded by epsilon = 10/255")
    axes[1].imshow(
        display,
        origin="lower",
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Normalized version\nfor visualization")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    try:
        build_fourier_basis(args.figure_dir / "theory" / "fourier_basis_2d_en.pdf")
        build_periodicity_and_gibbs(
            args.figure_dir / "theory" / "fourier_periodicite_gibbs_en.pdf"
        )
        build_mean_clean_spectrum(
            args.data_dir,
            args.figure_dir
            / "results"
            / "mean_clean_spectrum_comparison_cifar_en.pdf",
        )
        build_spatial_sfa_pattern(
            args.figure_dir / "results" / "sfa_spatial_pattern_vgg_cifar10_en.pdf"
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from None


if __name__ == "__main__":
    main()
