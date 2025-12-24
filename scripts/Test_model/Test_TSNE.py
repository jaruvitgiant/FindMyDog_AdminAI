import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def run_tsne(X, y, perplexity=30, n_iter=1000, random_state=42):
    """
    Run t-SNE on embeddings X with labels y and return matplotlib figure.
    X: numpy array (N, D)
    y: numpy array (N,)
    """

    X = np.asarray(X)
    y = np.asarray(y)

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(X) - 1),
        max_iter=n_iter,
        random_state=random_state,
        init="random",
        learning_rate="auto",
    )

    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))

    classes = np.unique(y)
    for c in classes:
        idx = y == c
        ax.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            label=str(c),
            s=20,
            alpha=0.7,
        )

    ax.set_title("t-SNE Visualization")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(markerscale=1.5, fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    return fig
