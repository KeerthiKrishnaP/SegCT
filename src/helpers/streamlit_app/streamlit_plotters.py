import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray


def plot_2d(x: NDArray, y: NDArray, xlabel: str, ylabel: str) -> Figure:
    """Generate and return a 2D heatmap with transparent scatter overlay."""
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    fig, ax = plt.subplots(figsize=(7, 6))

    # Heatmap (2D histogram)
    h = ax.hist2d(x, y, bins=50, cmap="plasma", alpha=0.6)

    # Transparent scatter overlay
    ax.scatter(x, y, color="None", s=1, alpha=0.1, edgecolors="black", linewidths=0.5)

    # Labels and colorbar
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("2D Heatmap with Transparent Scatter")
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label("Point Density")

    plt.tight_layout()

    return fig


def plot_3d(
    x: NDArray, y: NDArray, z: NDArray, xlabel: str, ylabel: str, zlabel: str
) -> Figure:
    """Generate 2D density projections of 3D data with scatter overlays."""
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    z = np.asarray(z).flatten()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # XY plane
    axs[0].hist2d(x, y, bins=50, cmap="plasma", alpha=0.6)
    axs[0].scatter(
        x, y, color="None", s=1, alpha=0.1, edgecolors="black", linewidths=0.5
    )
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[0].set_title("XY Projection")

    # XZ plane
    axs[1].hist2d(x, z, bins=50, cmap="plasma", alpha=0.6)
    axs[1].scatter(
        x, z, color="None", s=1, alpha=0.1, edgecolors="black", linewidths=0.5
    )
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(zlabel)
    axs[1].set_title("XZ Projection")

    # YZ plane
    axs[2].hist2d(y, z, bins=50, cmap="plasma", alpha=0.6)
    axs[2].scatter(
        y, z, color="None", s=1, alpha=0.1, edgecolors="black", linewidths=0.5
    )
    axs[2].set_xlabel(ylabel)
    axs[2].set_ylabel(zlabel)
    axs[2].set_title("YZ Projection")

    plt.tight_layout()

    return fig
