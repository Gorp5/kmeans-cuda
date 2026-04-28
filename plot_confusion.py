import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="dir with assignments.csv")
    ap.add_argument("--out", default="confusion.png")
    args = ap.parse_args()

    arr = np.loadtxt(Path(args.dir) / "assignments.csv",
                     delimiter=",", skiprows=1, dtype=np.int32)
    y, clusters = arr[:, 1], arr[:, 2]
    k, NC = clusters.max() + 1, 10

    # cluster -> digit by majority vote
    hist = np.zeros((k, NC), dtype=int)
    np.add.at(hist, (clusters, y), 1)
    cluster_to_digit = hist.argmax(axis=1)
    pred = cluster_to_digit[clusters]

    # confusion matrix + per-row normalization (so each row sums to 1)
    confusion = np.zeros((NC, NC), dtype=int)
    np.add.at(confusion, (y, pred), 1)
    norm = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)

    # metrics summary for the title
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    p  = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r  = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    acc = (pred == y).mean()

    # plot
    fig, ax = plt.subplots(figsize=(9, 8), facecolor="#0c0c0c")
    ax.set_facecolor("#0c0c0c")
    im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1, aspect="equal")

    # annotate each cell with the count and row-fraction
    for t in range(NC):
        for q in range(NC):
            v = confusion[t, q]
            if v == 0:
                continue
            color = "white" if norm[t, q] < 0.5 else "black"
            ax.text(q, t, f"{v}", ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_xticks(range(NC))
    ax.set_yticks(range(NC))
    ax.set_xticklabels(range(NC), color="white")
    ax.set_yticklabels(range(NC), color="white")
    ax.set_xlabel("Predicted digit (cluster majority vote)", color="white", fontsize=12)
    ax.set_ylabel("True digit", color="white", fontsize=12)
    ax.set_title(
        f"Confusion matrix  —  accuracy = {acc:.3f}  ·  macro-F1 = {f1.mean():.3f}",
        color="white", fontsize=13, pad=14,
    )
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.tick_params(colors="white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("row-normalized fraction", color="white")
    cbar.outline.set_edgecolor("#333")

    plt.tight_layout()
    out = Path(args.out)
    plt.savefig(out, dpi=120, facecolor="#0c0c0c")
    print(f"saved {out.resolve()}")
    print(f"accuracy = {acc:.4f}   macro-F1 = {f1.mean():.4f}")


if __name__ == "__main__":
    main()