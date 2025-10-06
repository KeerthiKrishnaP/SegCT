import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew

# ===============================================
# STEP 1 — Simulate Feature Histograms for Regions
# ===============================================
num_regions = 120
num_bins = 40
features = ["gray", "azimuth", "anisotropy"]

big_image_features = {f: np.random.rand(num_regions, num_bins) for f in features}
for f in features:
    big_image_features[f] /= np.sum(big_image_features[f], axis=1, keepdims=True)


# ===============================================
# STEP 2 — Compute Statistical Descriptors
# ===============================================
def extract_stats(hist):
    """Compute mean, std, skewness, kurtosis from histogram"""
    bins = np.linspace(0, 1, len(hist))
    mean = np.sum(bins * hist)
    std = np.sqrt(np.sum((bins - mean) ** 2 * hist))
    sk = np.sum(((bins - mean) ** 3) * hist) / (std**3 + 1e-12)
    ku = np.sum(((bins - mean) ** 4) * hist) / (std**4 + 1e-12)
    return np.array([mean, std, sk, ku])


X = []
for i in range(num_regions):
    row = []
    for f in features:
        row.extend(extract_stats(big_image_features[f][i]))
    X.append(row)
X = np.array(X)  # shape (num_regions, 12)


# ===============================================
# STEP 3 — Implement K-Means Clustering from Scratch
# ===============================================
def kmeans(X, k=4, max_iter=100, tol=1e-4):
    n_samples, n_features = X.shape
    np.random.seed(42)
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    for iteration in range(max_iter):
        # Assign each sample to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        # Update centroids
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        # Convergence check
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return labels, centroids


labels, centroids = kmeans(X, k=4)
print("K-Means clustering complete!")

# ===============================================
# STEP 4 — Visualize Results
# ===============================================
plt.figure(figsize=(8, 2))
plt.scatter(np.arange(num_regions), np.zeros(num_regions), c=labels, cmap="tab10", s=50)
plt.yticks([])
plt.title("Segmentation via K-Means Clustering (from scratch)")
plt.xlabel("Region Index")
plt.show()

# ===============================================
# STEP 5 — Visualize Cluster Means
# ===============================================
plt.figure(figsize=(6, 4))
for i in range(centroids.shape[0]):
    plt.plot(centroids[i], label=f"Cluster {i}")
plt.title("Cluster Mean Feature Profiles (12D compressed to 1D)")
plt.legend()
plt.show()
