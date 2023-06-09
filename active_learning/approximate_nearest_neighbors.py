import time
import sys
import pdb

try:
    import annoy
except ImportError:
    print("The package 'annoy' is required to run this example.")
    sys.exit()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsTransformer
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

class AnnoyTransformer(TransformerMixin, BaseEstimator):
    """Wrapper for using annoy.AnnoyIndex as sklearn's KNeighborsTransformer"""

    def __init__(self, n_neighbors=5, metric="angular", n_trees=10, search_k=-1):
        self.n_neighbors = n_neighbors
        self.n_trees = n_trees
        self.search_k = search_k
        self.metric = metric

    def fit(self, X):
        self.n_samples_fit_ = X.shape[0]
        # print("Building Annoy index with metric %s, %d trees, search_k=%d" % (self.metric, self.n_trees, self.search_k))
        self.annoy_ = annoy.AnnoyIndex(X.shape[1], metric=self.metric)
        # print("Adding %d vectors to Annoy index" % X.shape[0])
        for i, x in enumerate(X):
            # print("Adding vector %d to Annoy index" % i)
            self.annoy_.add_item(i, x.tolist())
        self.annoy_.build(self.n_trees)
        # print("Annoy index built!")
        return self

    def transform(self, X):
        return self._transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X)._transform(X=None)

    def _transform(self, X):
        """As `transform`, but handles X is None for faster `fit_transform`."""
        # pdb.set_trace()
        n_samples_transform = self.n_samples_fit_ if X is None else X.shape[0]

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        n_neighbors = self.n_neighbors + 1

        indices = np.empty((n_samples_transform, n_neighbors), dtype=int)
        distances = np.empty((n_samples_transform, n_neighbors))

        if X is None:
            for i in range(self.annoy_.get_n_items()):
                ind, dist = self.annoy_.get_nns_by_item(
                    i, n_neighbors, self.search_k, include_distances=True
                )

                indices[i], distances[i] = ind, dist
        else:
            for i, x in enumerate(X):
                indices[i], distances[i] = self.annoy_.get_nns_by_vector(
                    x.tolist(), n_neighbors, self.search_k, include_distances=True
                )

        indptr = np.arange(0, n_samples_transform * n_neighbors + 1, n_neighbors)
        kneighbors_graph = csr_matrix(
            (distances.ravel(), indices.ravel(), indptr),
            shape=(n_samples_transform, self.n_samples_fit_),
        )

        return kneighbors_graph


def test_transformers():
    """Test that AnnoyTransformer and KNeighborsTransformer give same results"""
    X = np.random.RandomState(42).randn(12, 2000)
    # print("made X")
    knn = KNeighborsTransformer(n_neighbors=X.shape[0]-1, metric="cosine")
    Xt0 = knn.fit_transform(X)
    print(Xt0)
    # pdb.set_trace()
    # print("made knn")
    ann = AnnoyTransformer(n_neighbors=X.shape[0]-1, metric="angular")
    # print("made annoy")
    Xt1 = ann.fit_transform(X)
    print(Xt1)
    # print("fit annoy")

def load_mnist(n_samples):
    """Load MNIST, shuffle the data, and return only n_samples."""
    mnist = fetch_openml("mnist_784", as_frame=False)
    X, y = shuffle(mnist.data, mnist.target, random_state=2)
    return X[:n_samples] / 255, y[:n_samples]


def run_benchmark():
    print("Benchmarking KNeighborsTransformer and AnnoyTransformer...")
    datasets = [
        ("MNIST_2000", load_mnist(n_samples=2000)),
        # ("MNIST_10000", load_mnist(n_samples=10000)),
    ]
    print("Datasets loaded!")
    perplexity = 30
    metric = "cosine"
    # TSNE requires a certain number of neighbors which depends on the
    # perplexity parameter.
    # Add one since we include each sample as its own neighbor.
    n_neighbors = int(3.0 * perplexity + 1) + 1
    transformers = [
        ("AnnoyTransformer", AnnoyTransformer(n_neighbors=n_neighbors, metric=metric)),
        (
            "KNeighborsTransformer",
            KNeighborsTransformer(
                n_neighbors=n_neighbors, mode="distance", metric=metric
            ),),]
    for dataset_name, (X, y) in datasets:
        msg = "Benchmarking on %s:" % dataset_name
        print("\n%s\n%s" % (msg, "-" * len(msg)))
        for transformer_name, transformer in transformers:
            start = time.time()
            Xt = transformer.fit_transform(X)
            duration = time.time() - start
            # print the duration report
            longest = np.max([len(name) for name, model in transformers])
            whitespaces = " " * (longest - len(transformer_name))
            print("%s: %s%.3f sec" % (transformer_name, whitespaces, duration))

if __name__ == "__main__":
    print("Testing annoy transformer...")
    test_transformers()
    print("annoy transformer's test done!\n")
    run_benchmark()