import argparse
import struct
import time
from pathlib import Path
import umap
import sklearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_mnist_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic} in {path}"
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    return raw.reshape(n, rows * cols).astype(np.float32) / 255.0


def load_assignments(path):
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int32)
    # each rows are image_id, true_label, cluster_id
    # in accesding order
    return arr[:, 1], arr[:, 2]

cmap = plt.get_cmap("tab10")
def scatter(ax, pts, color_by, title):
    ax.set_facecolor("#0c0c0c")
    ax.scatter(pts[:, 0], pts[:, 1], c=color_by, cmap=cmap, s=4, alpha=0.55, linewidths=0)
    ax.set_title(title, color="white", fontsize=14, pad=10)
    ax.set_xticks([]); ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_color("#333")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="directory with MNIST + assignments.csv")
    ap.add_argument("--out", default="cluster_projection.png")
    ap.add_argument("--umap-sample", type=int, default=15000, help="subsample size for UMAP (it's O(N log N) per epoch). " "PCA still uses all 60k.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = Path(args.dir)

    # Data loading phase
    X = load_mnist_images(d / "train-images-idx3-ubyte")
    print(f"  X: {X.shape}")

    y, clusters = load_assignments(d / "assignments.csv")
    assert len(y) == len(X), "assignments.csv row count != image count"
    print(f"  {len(y)} assignments, {clusters.max() + 1} clusters")

    y_pred = np.zeros_like(y)

    for c in np.unique(clusters):
        mask = (clusters == c)
        majority_label = np.bincount(y[mask]).argmax()
        y_pred[mask] = majority_label

    f1 = sklearn.metrics.f1_score(y, y_pred, average='macro')
    accuracy = sklearn.metrics.accuracy_score(y, y_pred)
    precision = sklearn.metrics.precision_score(y, y_pred, average='macro')

    print(f"Precision: {precision}\nAccuracy: {accuracy}\n F1: {f1}")

    # Computing using PCA
    t = time.time()
    X_pca = PCA(n_components=2, random_state=args.seed).fit_transform(X)

    # Subsampling
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X), size=args.umap_sample, replace=False)
    X50 = PCA(n_components=50, random_state=args.seed).fit_transform(X[idx])

    t = time.time()
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, n_epochs=200, random_state=args.seed, verbose=False)
    X_umap = reducer.fit_transform(X50)

    # Plotting starts here
    print("rendering...")
    pca_pts      = X_pca[idx]
    cluster_sub  = clusters[idx]
    y_sub        = y[idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 13), facecolor="#0c0c0c")
    fig.suptitle(f"MNIST: {args.umap_sample:,} points projected to 2D", fontsize=18, color="white", y=0.995)

    scatter(axes[0, 0], pca_pts, cluster_sub, "PCA  —  colored by k-means cluster")
    scatter(axes[0, 1], pca_pts, y_sub,       "PCA  —  colored by TRUE digit label")
    scatter(axes[1, 0], X_umap,  cluster_sub, "UMAP  —  colored by k-means cluster")
    scatter(axes[1, 1], X_umap,  y_sub,       "UMAP  —  colored by TRUE digit label")

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=cmap(i), markeredgecolor="none", markersize=10, label=f"{i}") for i in range(10)]
    fig.legend(handles=handles, loc="lower center", ncol=10,frameon=False, labelcolor="white", fontsize=12,bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    out = Path(args.out)
    plt.savefig(out, dpi=110, facecolor="#0c0c0c")
    print(f"saved {out.resolve()}")


if __name__ == "__main__":
    main()