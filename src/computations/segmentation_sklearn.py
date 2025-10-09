import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy, wasserstein_distance
from tqdm import tqdm


def segment_3d_image_from_data(
    training_data: dict,
    image_features: dict,
    window_size: int = 15,
    num_bins: int = 50,
    visualize: bool = False,
) -> np.ndarray:
    """
    Segments a 3D image into regions (matrix, warp, weft, void) based on
    statistical similarity between feature distributions.

    Parameters
    ----------
    training_data : dict
        Dictionary structured as:
        {
          "warp": {"gray": arr, "azimuth": arr, "anisotropy": arr},
          "weft": {...}, "matrix": {...}, "void": {...}
        }
        Each feature array is 1D (flattened values).
    image_features : dict
        Dictionary structured as:
        {"gray": 3D_array, "azimuth": 3D_array, "anisotropy": 3D_array}
        All arrays must have the same shape.
    window_size : int
        Cubic window size (e.g., 15 → 15×15×15 voxels per segment).
    num_bins : int
        Number of bins for feature histograms.
    visualize : bool
        If True, shows a mid-slice of the segmented volume.

    Returns
    -------
    segmented : np.ndarray
        3D array (downsampled by window size) of assigned class labels.
    """

    feature_names = list(image_features.keys())
    labels = list(training_data.keys())

    # ------------------------------------------------------------
    # 1. Compute class histograms from training data
    # ------------------------------------------------------------
    hist_class = {}
    for label in labels:
        hist_class[label] = {}
        for feature in feature_names:
            h, _ = np.histogram(
                training_data[label][feature], bins=num_bins, density=True
            )
            hist_class[label][feature] = h + 1e-12

    # ------------------------------------------------------------
    # 2. Distance metric (KL + Wasserstein)
    # ------------------------------------------------------------
    def compute_class_distance(region_hists, label):
        dists = []
        for f in feature_names:
            hist_r = region_hists[f] + 1e-12
            hist_c = hist_class[label][f]
            d_kl = entropy(hist_r, hist_c)
            d_ws = wasserstein_distance(hist_r, hist_c)
            dists.append(0.5 * d_kl + 0.5 * d_ws)
        return np.mean(dists)

    # ------------------------------------------------------------
    # 3. Segment the 3D image
    # ------------------------------------------------------------
    shape = image_features[feature_names[0]].shape
    nx, ny, nz = shape
    sx = sy = sz = window_size
    nx_s, ny_s, nz_s = nx // sx, ny // sy, nz // sz

    segmented = np.empty((nx_s, ny_s, nz_s), dtype="object")

    print("Segmenting volume...")
    for ix in tqdm(range(nx_s)):
        for iy in range(ny_s):
            for iz in range(nz_s):
                # Extract region for each feature
                region = {
                    f: image_features[f][
                        ix * sx : (ix + 1) * sx,
                        iy * sy : (iy + 1) * sy,
                        iz * sz : (iz + 1) * sz,
                    ].ravel()
                    for f in feature_names
                }

                # Compute normalized histograms
                region_hists = {}
                for f in feature_names:
                    h, _ = np.histogram(region[f], bins=num_bins, density=True)
                    region_hists[f] = h / (np.sum(h) + 1e-12)

                # Compare to each class
                distances = {
                    label: compute_class_distance(region_hists, label)
                    for label in labels
                }
                segmented[ix, iy, iz] = min(distances, key=distances.get)  # type: ignore

    print("Segmentation complete! Output shape:", segmented.shape)

    # ------------------------------------------------------------
    # 4. Visualization (optional)
    # ------------------------------------------------------------
    if visualize:
        mid_z = segmented.shape[2] // 2
        plt.figure(figsize=(6, 5))
        plt.imshow(pd.Categorical(segmented[:, :, mid_z]).codes, cmap="tab10")
        plt.title("Segmented Volume (mid slice)")
        plt.axis("off")
        plt.show()

    return segmented
