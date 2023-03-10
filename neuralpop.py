"""
Neural population analysis
"""


import numpy as np
from sklearn.mixture import GaussianMixture
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class Clusters:
    """
    Container for clustering results.
    Cluster labels are remapped to correspond with the rank of the centroid norm.
    """

    def __init__(self, gmm: GaussianMixture, labels: np.ndarray):

        sorted_ixs = list(np.argsort(np.linalg.norm(gmm.means_, axis=1)))
        sorted_labels = np.zeros_like(labels)
        for label in range(gmm.n_components):
            sorted_labels[labels == label] = sorted_ixs.index(label)

        self.k = gmm.n_components
        self.labels = sorted_labels
        self.means = gmm.means_[sorted_ixs].squeeze()
        self.covars = gmm.covariances_[sorted_ixs].squeeze()
        self.sizes = [np.sum(self.labels == label) for label in range(self.k)]

    def get_elements(self, label):
        """ get elements of cluster. supports negative indexing. """
        if label < 0:
            label = self.k + label
        return np.nonzero(self.labels == label)[0]


def gaussian_mixture(x, ks, verbose=True, gmm_kw=None):

    x = np.array(x)
    if np.ndim(x) == 1:
        x = x[:, None]

    if isinstance(ks, int):
        ks = 1 + np.arange(ks)

    if gmm_kw is None:
        gmm_kw = {}

    best = {}
    bics, aics = [], []
    for k in ks:
        gmm = GaussianMixture(n_components=k, **gmm_kw)
        gmm.fit(x)
        bics.append(gmm.bic(x))
        aics.append(gmm.aic(x))
        if (not best) or (bics[-1] < best["bic"]):
            best = {"gmm": gmm, "bic": bics[-1]}
    clusters = Clusters(best["gmm"], best["gmm"].predict(x))

    if verbose:
        bic_k = ks[np.argmin(bics)]
        aic_k = ks[np.argmin(aics)]
        if len(ks) > 1:
            print(f"Number of populations: By BIC={bic_k}, By AIC={aic_k}, Chosen={clusters.k}. ", end="")
        else:
            print(f"Number of populations: {clusters.k}. ", end="")
        for label in range(clusters.k):
            print("#{:d}: N={:d}, Mean={:2.3f}, SD={:2.3f}. ".format(
                label + 1, clusters.sizes[label], clusters.means[label], clusters.covars[label] ** .5), end="")
        print("")

    return clusters, ks, bics, aics


def calc_neural_populations(model, neuron_ixs, k=None):

    neuron_weight = {ix: np.mean(np.abs(model.coef_[neuron_ixs == ix]))
                     for ix in set(neuron_ixs)}

    clusters, ks, bics, aics = gaussian_mixture(
        x=list(neuron_weight.values()),
        ks=9 if k is None else [k],
        verbose=True,
        gmm_kw={"random_state": 0, "n_init": 10})

    return {
        "neuron_weight": neuron_weight,
        "weightsorted_neurons": np.argsort(list(neuron_weight.values()))[::-1],
        "ks": ks,
        "bics": bics,
        "aics": aics,
        "clusters": clusters
    }


def show_num_cluster_selection(ks, bics, aics, chosen_k=None):
    plt.figure(figsize=(4, 3))
    plt.plot(ks, bics, 'ks:')
    plt.plot(ks, aics, 'ko:')
    plt.xlabel('# components')
    plt.xticks(ks)
    plt.legend(['BIC', 'AIC'])
    if chosen_k is not None:
        plt.plot(chosen_k, bics[list(ks).index(chosen_k)], 'rs')
        plt.plot(chosen_k, aics[list(ks).index(chosen_k)], 'ro')
    plt.title('BIC and AIC vs #Clusters')
