import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection


def plot_feature_relationships(phi, beta, gray, sample_size=200000) -> None:
    """
    Plot 3 two-feature histograms and one 3D scatter plot for φ, β, and intensity.

    Parameters
    ----------
    phi : np.ndarray
        Azimuthal angle field (same shape as beta and gray).
    beta : np.ndarray
        Structural anisotropy field.
    gray : np.ndarray
        Gray value or intensity field.
    sample_size : int
        Number of random voxels to sample for plotting (useful for large data).
    """

    # Flatten and sample to avoid plotting millions of points
    phi_flat = phi.ravel()
    beta_flat = beta.ravel()
    gray_flat = gray.ravel()

    n_points = phi_flat.size
    if n_points > sample_size:
        idx = np.random.choice(n_points, size=sample_size, replace=False)
        phi_flat = phi_flat[idx]
        beta_flat = beta_flat[idx]
        gray_flat = gray_flat[idx]

    # --- Create figure with 4 subplots ---
    fig = plt.figure(figsize=(18, 12))

    # β vs intensity
    ax1 = fig.add_subplot(2, 2, 1)
    hb1 = ax1.hexbin(beta_flat, gray_flat, gridsize=80, cmap="viridis", bins="log")
    ax1.set_xlabel("Anisotropy β")
    ax1.set_ylabel("Gray intensity")
    ax1.set_title("β vs Intensity")
    fig.colorbar(hb1, ax=ax1, label="log(count)")

    # φ vs β
    ax2 = fig.add_subplot(2, 2, 2)
    hb2 = ax2.hexbin(phi_flat, beta_flat, gridsize=80, cmap="viridis", bins="log")
    ax2.set_xlabel("Azimuthal angle φ [rad]")
    ax2.set_ylabel("Anisotropy β")
    ax2.set_title("φ vs β")
    fig.colorbar(hb2, ax=ax2, label="log(count)")

    # φ vs intensity
    ax3 = fig.add_subplot(2, 2, 3)
    hb3 = ax3.hexbin(phi_flat, gray_flat, gridsize=80, cmap="viridis", bins="log")
    ax3.set_xlabel("Azimuthal angle φ [rad]")
    ax3.set_ylabel("Gray intensity")
    ax3.set_title("φ vs Intensity")
    fig.colorbar(hb3, ax=ax3, label="log(count)")

    # 3D scatter: φ, β, intensity
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    sc = ax4.scatter(
        phi_flat, beta_flat, gray_flat, c=gray_flat, cmap="viridis", alpha=0.5
    )
    ax4.set_xlabel("φ [rad]")
    ax4.set_ylabel("β")
    ax4.set_zlabel("Gray intensity")  # type: ignore[attr-defined]
    ax4.set_title("3D feature space: (φ, β, Intensity)")
    fig.colorbar(sc, ax=ax4, shrink=0.6, label="Gray intensity")

    plt.tight_layout()
    plt.show()
