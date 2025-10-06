import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz
from scipy.stats import entropy, ks_2samp

# ============================================================
# STEP 1 — Simulated or Loaded Training Data (Distributions)
# ============================================================

# Each class contains raw distributions (e.g., from .npy or measurements)
training_data = {
    "warp": {
        "gray": np.random.normal(0.55, 0.1, 800),
        "azimuth": np.random.normal(45, 10, 800),
        "anisotropy": np.random.normal(0.8, 0.05, 800),
    },
    "weft": {
        "gray": np.random.normal(0.65, 0.1, 800),
        "azimuth": np.random.normal(135, 15, 800),
        "anisotropy": np.random.normal(0.7, 0.05, 800),
    },
    "matrix": {
        "gray": np.random.normal(0.35, 0.1, 800),
        "azimuth": np.random.normal(90, 12, 800),
        "anisotropy": np.random.normal(0.5, 0.08, 800),
    },
    "void": {
        "gray": np.random.normal(0.1, 0.05, 800),
        "azimuth": np.random.normal(0, 5, 800),
        "anisotropy": np.random.normal(0.2, 0.05, 800),
    },
}

feature_names = ["gray", "azimuth", "anisotropy"]
num_bins = 50

# ============================================================
# STEP 2 — Compute Normalized Histograms for Training Classes
# ============================================================


def compute_histograms(training_data, num_bins=50):
    """Convert raw feature arrays into normalized histograms."""
    histograms = {}
    for cls, feats in training_data.items():
        histograms[cls] = {}
        for feat_name, values in feats.items():
            hist, edges = np.histogram(values, bins=num_bins, density=True)
            histograms[cls][feat_name] = hist / (np.sum(hist) + 1e-12)
    return histograms


training_hists = compute_histograms(training_data, num_bins)

# ============================================================
# STEP 3 — Generate (or Load) Feature Distributions for Big Image
# ============================================================

num_regions = 100  # number of regions to classify
big_image_features = {f: np.random.rand(num_regions, num_bins) for f in feature_names}
# Normalize each histogram per region
for f in feature_names:
    big_image_features[f] /= (
        np.sum(big_image_features[f], axis=1, keepdims=True) + 1e-12
    )

# ============================================================
# STEP 4 — Define Statistical Distance Functions
# ============================================================


def kl_divergence(p, q):
    p = p + 1e-12
    q = q + 1e-12
    return np.sum(p * np.log(p / q))


def wasserstein_1d(p, q):
    """Approximate 1D Wasserstein distance between two histograms."""
    cdf_p = np.cumsum(p) / np.sum(p)
    cdf_q = np.cumsum(q) / np.sum(q)
    return trapz(np.abs(cdf_p - cdf_q))


def ks_distance(p, q):
    """Kolmogorov-Smirnov distance between two histograms."""
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return np.max(np.abs(cdf_p - cdf_q))


def composite_distance(p, q):
    """Weighted combination of multiple distances."""
    d1 = kl_divergence(p, q)
    d2 = wasserstein_1d(p, q)
    d3 = ks_distance(p, q)
    return 0.4 * d1 + 0.4 * d2 + 0.2 * d3


# ============================================================
# STEP 5 — Assign Each Region to the Closest Class
# ============================================================

labels = list(training_hists.keys())
region_labels = []

for i in range(num_regions):
    distances = {}
    for cls in labels:
        total_d = 0
        for feat in feature_names:
            total_d += composite_distance(
                big_image_features[feat][i], training_hists[cls][feat]
            )
        distances[cls] = total_d / len(feature_names)
    region_labels.append(min(distances, key=distances.get))

region_labels = np.array(region_labels)
print("Segmentation complete! Example:", region_labels[:10])

# ============================================================
# STEP 6 — Visualize Segmentation Map
# ============================================================

plt.figure(figsize=(8, 2))
plt.scatter(
    np.arange(num_regions),
    np.zeros(num_regions),
    c=[labels.index(l) for l in region_labels],
    cmap="tab10",
    s=40,
)
plt.yticks([])
plt.title("Segmented Regions by Class (from scratch)")
plt.xlabel("Region Index")
plt.show()

# ============================================================
# STEP 7 — Visualize Feature Distributions per Class
# ============================================================

plt.figure(figsize=(12, 4))
for i, feat in enumerate(feature_names):
    plt.subplot(1, 3, i + 1)
    for cls in labels:
        vals = training_data[cls][feat]
        plt.hist(vals, bins=30, alpha=0.5, density=True, label=cls)
    plt.title(feat)
    plt.legend()
plt.suptitle("Training Class Feature Distributions")
plt.tight_layout()
plt.show()

# ============================================================
# STEP 8 — Quantify Class Separation
# ============================================================

print("\nAverage Pairwise Class Distances:")
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        cls1, cls2 = labels[i], labels[j]
        d_sum = 0
        for feat in feature_names:
            d_sum += composite_distance(
                training_hists[cls1][feat], training_hists[cls2][feat]
            )
        print(f"{cls1} vs {cls2} : {d_sum / len(feature_names):.4f}")
