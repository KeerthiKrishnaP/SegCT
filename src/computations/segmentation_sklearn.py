import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import entropy, ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================
# STEP 1 — Load Training Data (Distributions)
# ============================================================

# Example structure:
# training_data = {
#     "warp": {"gray": np.load("warp_gray.npy"),
#               "azimuth": np.load("warp_azimuth.npy"),
#               "anisotropy": np.load("warp_aniso.npy")},
#     "weft": {...},
#     ...
# }

training_data = {
    "warp": {
        "gray": np.random.normal(0.5, 0.1, 500),
        "azimuth": np.random.normal(45, 10, 500),
        "anisotropy": np.random.normal(0.8, 0.05, 500),
    },
    "weft": {
        "gray": np.random.normal(0.6, 0.1, 500),
        "azimuth": np.random.normal(135, 10, 500),
        "anisotropy": np.random.normal(0.7, 0.05, 500),
    },
    "matrix": {
        "gray": np.random.normal(0.3, 0.1, 500),
        "azimuth": np.random.normal(90, 10, 500),
        "anisotropy": np.random.normal(0.5, 0.1, 500),
    },
    "void": {
        "gray": np.random.normal(0.1, 0.05, 500),
        "azimuth": np.random.normal(0, 5, 500),
        "anisotropy": np.random.normal(0.2, 0.05, 500),
    },
}

# ============================================================
# STEP 2 — Load Feature Distributions for Each Region/Pixel in Big Image
# ============================================================

# Suppose big_image_features is an array of shape (num_regions, num_bins, num_features)
# representing histograms for each region
# For demo, we simulate this

num_regions = 100
num_bins = 50
feature_names = ["gray", "azimuth", "anisotropy"]

big_image_features = {
    name: np.random.rand(num_regions, num_bins) for name in feature_names
}
for name in feature_names:
    # Normalize histograms to sum to 1
    big_image_features[name] /= big_image_features[name].sum(axis=1, keepdims=True)

# ============================================================
# STEP 3 — Define Statistical Distance Functions
# ============================================================


def compute_class_distance(region_hist, class_hist):
    """Compute mean distance over all features between region and class distributions"""
    dists = []
    for feature in feature_names:
        # Convert training distribution into histogram
        hist_class, _ = np.histogram(class_hist[feature], bins=num_bins, density=True)
        hist_class += 1e-12  # avoid log(0)
        hist_region = region_hist[feature]
        hist_region += 1e-12
        # Compute distances (you can pick one or combine)
        d_kl = entropy(hist_region, hist_class)
        d_ws = wasserstein_distance(hist_region, hist_class)
        dists.append(0.5 * d_kl + 0.5 * d_ws)
    return np.mean(dists)


# ============================================================
# STEP 4 — Segment: Assign Each Region to the Closest Class
# ============================================================

labels = list(training_data.keys())
segmentation = []

for i in range(num_regions):
    region_hist = {name: big_image_features[name][i, :] for name in feature_names}
    d = {
        label: compute_class_distance(region_hist, training_data[label])
        for label in labels
    }
    assigned_label = min(d, key=d.get)
    segmentation.append(assigned_label)

segmentation = np.array(segmentation)
print("Segmentation complete!")

# ============================================================
# STEP 5 — Visualization of Segmentation Map
# ============================================================

plt.figure(figsize=(6, 2))
plt.scatter(
    np.arange(num_regions),
    np.zeros(num_regions),
    c=pd.Categorical(segmentation).codes,
    cmap="tab10",
    s=40,
)
plt.yticks([])
plt.xlabel("Region Index")
plt.title("Segmented Regions by Class")
plt.show()

# ============================================================
# STEP 6 — Compare Feature Distributions Between Classes
# ============================================================

plt.figure(figsize=(12, 4))
for i, feature in enumerate(feature_names):
    plt.subplot(1, 3, i + 1)
    for label in labels:
        plt.hist(
            training_data[label][feature], bins=30, alpha=0.5, label=label, density=True
        )
    plt.title(feature)
    plt.legend()
plt.suptitle("Class-wise Feature Distributions")
plt.tight_layout()
plt.show()

# ============================================================
# STEP 7 — Optional: Visualize Class Separation in Feature Space (Mean Stats)
# ============================================================

# Extract class mean feature vectors
class_vectors = []
class_labels = []
for label in labels:
    mean_vec = [np.mean(training_data[label][f]) for f in feature_names]
    class_vectors.append(mean_vec)
    class_labels.append(label)

class_vectors = np.array(class_vectors)
scaler = StandardScaler()
pca = PCA(n_components=3)
proj = pca.fit_transform(scaler.fit_transform(class_vectors))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=range(len(labels)))
for i, label in enumerate(labels):
    ax.text(proj[i, 0], proj[i, 1], proj[i, 2], label)
ax.set_title("Class Separation in PCA Feature Space")
plt.show()

# ============================================================
# STEP 8 — Optional Evaluation (if you have ground truth)
# ============================================================

# Suppose you have true_labels for the big image regions:
# true_labels = np.load("big_image_labels.npy")
# from sklearn.metrics import classification_report
# print(classification_report(true_labels, segmentation))
