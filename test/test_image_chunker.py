import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.computations.comput_features import fast_eigen_computations, structural_tensor


@pytest.mark.parametrize(
    "case", ["flat", "plane_z", "gradient_x", "gaussian", "cylinder"]
)
def test_structural_tensor_cases(case):
    """Unit test for 3D structural tensor correctness under synthetic conditions."""
    # --- 1️⃣ Create synthetic test volume ---
    shape = (32, 32, 32)
    if case == "flat":
        image = np.ones(shape, dtype=np.float32)

    elif case == "plane_z":
        image = np.zeros(shape, dtype=np.float32)
        image[16:, :, :] = 1.0  # step along Z

    elif case == "gradient_x":
        x = np.linspace(0, 1, shape[2], dtype=np.float32)
        image = np.tile(x, (shape[0], shape[1], 1))

    elif case == "gaussian":
        X, Y, Z = np.meshgrid(
            np.linspace(-1, 1, shape[2]),
            np.linspace(-1, 1, shape[1]),
            np.linspace(-1, 1, shape[0]),
            indexing="ij",
        )
        image = np.exp(-(X**2 + Y**2 + Z**2) / 0.2).astype(np.float32)

    elif case == "cylinder":
        X, Y, Z = np.meshgrid(
            np.linspace(-1, 1, 64),
            np.linspace(-1, 1, 64),
            np.linspace(-1, 1, 64),
            indexing="ij",
        )
        r = np.sqrt(X**2 + Y**2)
        image = np.exp(-((r - 0.3) ** 2) / 0.02).astype(np.float32)

    else:
        raise ValueError(f"Unknown case: {case}")

    # --- Compute structural tensor and features ---
    tensor = structural_tensor(image, window_radius=2)
    features = fast_eigen_computations(tensor)

    λ1, λ2, λ3 = features["lambda1"], features["lambda2"], features["lambda3"]
    anisotropy = features["anisotropy"]

    # --- General assertions ---
    assert all(k in tensor for k in ["S11", "S22", "S33", "S12", "S13", "S23"]), (
        "Missing tensor components."
    )
    assert (
        np.allclose(tensor["S12"], tensor["S12"].T, atol=1e-6) is False
    )  # Ensure it's not trivially symmetric in memory
    assert np.all(np.isfinite(λ1)), "NaNs found in eigenvalues."
    assert np.all(λ1 >= λ2) and np.all(λ2 >= λ3), "Eigenvalue ordering failed."

    # --- 4️⃣ Case-specific expectations ---
    if case == "flat":
        assert np.allclose(λ1, 0, atol=1e-5)
        assert np.allclose(anisotropy, 0, atol=1e-5)

    elif case == "plane_z":
        assert np.mean(tensor["S33"]) > 10 * np.mean(tensor["S11"])
        assert 0.8 < np.nanmax(anisotropy) <= 1.0

    elif case == "gradient_x":
        assert np.mean(tensor["S11"]) > 10 * np.mean(tensor["S22"])
        assert 0.8 < np.nanmean(anisotropy) <= 1.0

    elif case == "gaussian":
        assert np.nanmean(anisotropy) < 0.3  # isotropic near center

    elif case == "cylinder":
        assert np.nanmax(anisotropy) > 0.7  # strong anisotropy at surface

    # --- Optional visualization (for manual inspection) ---
    if False:  # set True to debug visually
        z = image.shape[0] // 2
        fig, ax = plt.subplots(1, 3, figsize=(10, 4))
        ax[0].imshow(image[z], cmap="gray")
        ax[0].set_title(f"{case} input (Z={z})")
        ax[1].imshow(tensor["S11"][z], cmap="inferno")
        ax[1].set_title("S11")
        ax[2].imshow(anisotropy[z], cmap="viridis")
        ax[2].set_title("Anisotropy")
        plt.tight_layout()
        plt.show()
