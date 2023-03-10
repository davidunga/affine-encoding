import os
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from segments_factory import load_segments
from scipy.ndimage import convolve1d
from scipy import stats
from sklearn import linear_model
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, ParameterGrid
from glob import glob


models_dir = "./models"


def mean_filter1d(x, w: int, axis=-1):
    return convolve1d(x, np.ones(w), axis=axis) / w


def zscore(xs: np.ndarray, x: np.ndarray | float = None, axis=-1, robust=False):
    """
    zscore(xs) - standardize xs.
    zscore(xs, x) - zscore of x relative to xs population. x has to be either a scalar, or same size as xs.
    zscore(__, robust=True) - compute robust zscore based on median absolute deviation
    """

    if robust:
        loc = np.median(xs, axis=axis, keepdims=True)
        scale = 1.4826 * np.median(np.abs(loc - xs), axis=axis, keepdims=True)
    else:
        loc = np.mean(xs, axis=axis, keepdims=True)
        scale = np.std(xs, axis=axis, keepdims=True)

    if x is None:
        return (xs - loc) / np.maximum(scale, np.finfo(float).eps)
    return (x - loc) / np.maximum(scale, np.finfo(float).eps)


def normalize(x: np.ndarray, axis=-1, kind="std"):
    """
    normalize array x along axis. normalization kinds:
        "std"       - zero mean, unit std
        "mad"       - zero median, unit median absolute deviation
        "max"       - min=0, max=1
        "sum"       - unit sum
        "sumabs"    - unit sum of absolutes
        "norm"      - unit l2 norm
    """

    match kind:
        case "std":
            return zscore(x, axis=axis, robust=False)
        case "mad":
            return zscore(x, axis=axis, robust=True)
        case "max":
            mn = np.min(x, axis=axis, keepdims=True)
            return (x - mn) / (np.max(x, axis=axis, keepdims=True) - mn)
        case "sum":
            return x / np.sum(x, axis=axis, keepdims=True)
        case "sumabs":
            return x / np.sum(np.abs(x), axis=axis, keepdims=True)
        case "norm":
            return x / np.linalg.norm(axis=axis, keepdims=True)
        case _:
            raise ValueError("Unknown normalization kind " + (kind if isinstance(kind, str) else ""))


class SegmentData:

    def __init__(self, seg_file, train_size, neural_reduce_factor, kin_reduce,
                 kin_inliers_pcntl, normalize_neural, rand_seed, partic_neurons):

        self.segs, _, _ = load_segments(seg_file)
        self.train_size = train_size
        self._train_msk = None
        self.full_neuron_ixs = None
        self.neural_reduce_factor = neural_reduce_factor
        self.normalize_neural = normalize_neural
        self.x = None
        self.kin_reduce = kin_reduce
        self.kin_inliers_pcntl = kin_inliers_pcntl
        self.rand_seed = rand_seed
        self.partic_neurons = partic_neurons

        self._make_train_test_split()
        self._make_neural()

    @property
    def neuron_ixs(self):
        return self.full_neuron_ixs[self._partic_mask()]

    def _partic_mask(self):
        return np.array([ix in self.partic_neurons for ix in self.full_neuron_ixs])

    def get_data(self, kinvar):
        rand_state = np.random.get_state()
        np.random.seed(self.rand_seed)

        def _make_target(kinvar):
            y = np.array([s.kin[kinvar.replace("_shuff", "")] for s in self.segs])
            match self.kin_reduce:
                case "median":
                    y = np.median(y, axis=1)
                case "mean":
                    y = np.mean(y, axis=1)
                case _:
                    raise ValueError("Unknown kinematic reduce")
            if kinvar.endswith("_shuff"):
                np.random.shuffle(y)
            return y

        y = _make_target(kinvar)
        x = self.x[:, self._partic_mask()].copy()

        # remove samples with outlier target values:
        ymin, ymax = np.percentile(y[self._train_msk], self.kin_inliers_pcntl)
        ii = (y >= ymin) & (y <= ymax)
        y = y[ii]
        x = x[ii]
        train_msk = self._train_msk[ii]

        # normalize target values:
        y = zscore(y[train_msk], y, axis=0, robust=True)

        np.random.set_state(rand_state)
        return x[train_msk], y[train_msk], x[~train_msk], y[~train_msk]

    def _make_train_test_split(self):
        rand_state = np.random.get_state()
        np.random.seed(self.rand_seed)
        trials = [s.trial_ix for s in self.segs]
        n_trials = len(set(trials))
        n_train_trials = int(.5 + self.train_size * n_trials)
        train_trial_ixs = np.random.permutation(n_trials)[:n_train_trials]
        self._train_msk = np.array([s.trial_ix in train_trial_ixs for s in self.segs])
        assert abs(self._train_msk.mean() - self.train_size) < .05
        np.random.set_state(rand_state)

    def _make_neural(self):
        rand_state = np.random.get_state()
        np.random.seed(self.rand_seed)

        S_stack = np.stack([s.S for s in self.segs], axis=2)
        if self.neural_reduce_factor > 0:
            starts = np.arange(0, S_stack.shape[1], self.neural_reduce_factor)
            stops = starts + self.neural_reduce_factor
            assert stops[-1] >= S_stack.shape[1]
            stops[-1] = S_stack.shape[1]
            S_stack = np.stack([np.mean(S_stack[:, start: stop, :], axis=1)
                                for start, stop in zip(starts, stops)], axis=1)

        self.full_neuron_ixs = np.repeat(np.arange(S_stack.shape[0]), S_stack.shape[1])
        self.x = S_stack.reshape(-1, S_stack.shape[2]).T

        sd = np.std(self.x, axis=0)
        sd_eps = 1e-6 * sd.max()
        novar_cols = sd < sd_eps
        self.x[:, novar_cols] += sd_eps * np.random.standard_normal(self.x[:, novar_cols].shape)

        if self.normalize_neural != "none":
            self.x = normalize(self.x, axis=0, kind=self.normalize_neural)

        if isinstance(self.partic_neurons, int) and self.partic_neurons == -1:
            self.partic_neurons = list(set(self.full_neuron_ixs))
        assert hasattr(self.partic_neurons, "__len__")

        np.random.set_state(rand_state)


def cv_eval(model, x, y, folds, zscore_shuffs):
    kf = KFold(n_splits=folds)
    zscores = {'r2': [], 'rmse': [], 'mape': []}
    for train_ixs, test_ixs in kf.split(x):
        model.fit(x[train_ixs], y[train_ixs])
        y_pred = model.predict(x[test_ixs])
        y_true = y[test_ixs]
        shuff_scores = {k: [] for k in zscores}
        for _ in range(zscore_shuffs + 1):
            shuff_scores['r2'].append(metrics.r2_score(y_true=y_true, y_pred=y_pred))
            shuff_scores['rmse'].append(np.sqrt(metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)))
            shuff_scores['mape'].append(metrics.mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred))
            np.random.shuffle(y_true)
        for mtrc, vals in shuff_scores.items():
            zscores[mtrc].append(zscore(xs=vals[1:], x=vals[0], robust=False))

    return {k: np.array(v) for k, v in zscores.items()}


higher_is_better = {
    "r2": True,
    "rmse": False,
    "mape": False
}


def cv_fit_models(cfg, kinvars=None, dump=False):

    if kinvars is None:
        #kinvars = ["spd2", "spd0", "spd1", "velx", "vely", "acc2", "accx", "accy"]
        kinvars = ["spd2", "spd0", "acc2"]
        #kinvars = ["spd2", "spd0", "velx", "vely", "acc2", "accx", "accy"]

    seg_data = SegmentData(**cfg["data"])

    obj = {
        "cfg": cfg,
        "neuron_ixs": seg_data.neuron_ixs,
        "fits": {}
    }

    for kinvar in kinvars:
        cv_lowest_err = None
        cv_best = {}
        print("-- " + kinvar + ": ")
        x, y, _, _ = seg_data.get_data(kinvar)
        for cv_params in ParameterGrid(cfg['cv_grid']):
            #model = linear_model.Lasso(**cv_params)
            model = linear_model.LinearRegression(**cv_params)
            cv_zscores = cv_eval(model, x, y, folds=cfg['cv_folds'], zscore_shuffs=cfg['zscore_shuffs'])
            cv_err = cv_zscores[cfg['cv_metric']].mean() * (-1 if higher_is_better[cfg['cv_metric']] else 1)
            if (cv_lowest_err is None) or (cv_err < cv_lowest_err):
                cv_lowest_err = cv_err
                cv_best = {"params": cv_params, "zscores": cv_zscores}
        model = linear_model.LinearRegression(**cv_best["params"])
        #model = linear_model.Lasso(**cv_best["params"])
        model.fit(x, y)
        obj["fits"][kinvar] = {"model": model, "zscores": cv_best["zscores"]}
        print(" chosen: ", {k: model.get_params()[k] for k, v in cfg['cv_grid'].items() if len(v) > 1})
        for mtrc in ("r2", "rmse"):
            val = cv_best["zscores"][mtrc]
            print(f" {mtrc.upper():<6s} avg zscore: {val.mean():2.3f}")

    if dump:
        run_name = datetime.now().strftime("%m%d%Y-%H%M%S")
        pkl = models_dir + f"/{run_name}.pkl"
        pickle.dump(obj, open(pkl, 'wb'))
        print("Dumped: " + pkl)

    return obj


from dataclasses import dataclass


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


def is_affine(kinvar):
    return kinvar in ("spd0", "spd1", "crv0", "crv1")


def plot_tuning(results):
    import pandas as pd
    import seaborn as sns

    GEOM_PALETTE = {'FuAff': 'red',
                    'EqAff': 'springgreen',
                    'Eucld': 'dodgerblue'}

    df = pd.DataFrame(results)
    df = df.sort_values(by='kinvar', ascending=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='zscore', y='kinvar', data=df, hue='geom', palette=GEOM_PALETTE, dodge=False, errwidth=1)
    xticks = np.arange(0, 15)
    #plt.grid()
    plt.xlim(xticks[0], xticks[-1])
    #xticks = np.arange(np.round(df['zscore'].min()), np.round(df['zscore'].max() + 1)).astype(int)
    plt.xticks(ticks=xticks, labels=xticks)
    plt.legend(loc='upper right')
    plt.xlabel('R2 zscore')

    #plt.gca().yaxis.tick_right()
    #plt.gca().yaxis.set_label_position("right")


def analyze_models(pkl):
    obj = pickle.load(open(pkl, 'rb'))

    print("-- Calculating populations")
    populations = {}
    for kinvar in obj['fits']:
        print(" " + kinvar + ": ", end="")
        populations[kinvar] = calc_neural_populations(obj['fits'][kinvar]['model'], obj['neuron_ixs'],
                                                      2 if is_affine(kinvar) else 2)

    def _plot_tuning_scores(fit_res):
        res = []
        for kinvar in fit_res:
            for zscore in fit_res[kinvar]["zscores"]["r2"].squeeze():
                if kinvar == 'spd0':
                    vr = 'AffSpeed'
                elif kinvar == 'spd2':
                    vr = 'EucSpeed'
                elif kinvar == 'acc2':
                    vr = 'EucAccel'
                else:
                    raise ValueError()

                res.append({'kinvar': vr, 'zscore': zscore, 'geom': 'FuAff' if kinvar == 'spd0' else 'Eucld'})
        plot_tuning(res)
        plt.grid(True)
        plt.gca().set_axisbelow(True)

    _plot_tuning_scores(obj['fits'])
    plt.title("Neural Tuning Scores - Full Population")

    print("-- Full population zscores:")
    for kinvar in populations:
        print(kinvar + ":")
        for mtrc in ("r2", "rmse"):
            val = obj["fits"][kinvar]["zscores"][mtrc]
            print(f" {mtrc.upper():<6s} avg zscore: {val.mean():2.3f}")

    euclidean_neurons = []
    for kinvar in populations:
        if not is_affine(kinvar):
            euclidean_neurons += list(populations[kinvar]['weightsorted_neurons'][:5])

    for kinvar in populations:
        if not is_affine(kinvar):
            continue

        relevant_pop_sz = 5
        relevant_neurons = [n for n in populations[kinvar]['weightsorted_neurons']
                            if n not in euclidean_neurons][:relevant_pop_sz]

        cfg = obj['cfg']
        cfg['data']['partic_neurons'] = np.array(list(relevant_neurons))

        print(f"-- Fit with {kinvar.upper()} subpopulation ({len(relevant_neurons)} neurons):")
        subpop_obj = cv_fit_models(cfg, dump=False)
        print(f"-- End of fit with " + kinvar + "\n")

        _plot_tuning_scores(subpop_obj['fits'])
        plt.title("Neural Tuning Scores - Affine Population")
        plt.show()


    # kinvars = np.array(list(populations.keys()))
    # kinvars = np.array(['spd2', 'acc2', 'spd0'])
    # w = np.array([list(populations[kinvar]['neuron_weight'].values()) for kinvar in kinvars])
    # #w = normalize(w, axis=1, kind="sum")
    # r = w.shape[1] - stats.rankdata(w, axis=1)
    # r_spd0 = r[kinvars == "spd0"].squeeze()
    # r_other = np.min(r[kinvars != "spd0"], axis=0)
    #
    # print("-- Calculating fit using exclusively tuned population")
    # for kinvar in populations:
    #     print(" " + kinvar + ": ", end="")
    #     relevant_neurons = set(populations[kinvar]['clusters'].get_elements(-1))
    #     print(f" tuned={len(relevant_neurons)} ", end="")
    #     for other_kinvar in populations:
    #         if "spd0" not in (kinvar, other_kinvar):
    #             continue
    #         if other_kinvar != kinvar:
    #             other_neurons = populations[other_kinvar]['clusters'].get_elements(-1)
    #             relevant_neurons.difference_update(other_neurons)
    #     print(f", exclusive={len(relevant_neurons)}.")
    #     print(" Full population score:")
    #     for mtrc, val in obj["fits"][kinvar]["zscores"].items():
    #         print(f" {mtrc.upper():<6s} avg zscore: {val.mean():2.3f}")
    #     cfg = obj['cfg']
    #     cfg['data']['partic_neurons'] = np.array(list(relevant_neurons))
    #     cv_fit_models(cfg, dump=False)
    #     print(f" -- End of fit with exclusive {kinvar} neurons\n")


default_cfg = {

    "data": {
        "seg_file": "TP_RS bin10 lag100 dur200 - segments.json",
        "rand_seed": 0,
        "partic_neurons": -1,
        "train_size": 1,
        "normalize_neural": "std",
        "neural_reduce_factor": 5,
        "kin_reduce": "median",
        "kin_inliers_pcntl": [0, 95]
    },

    "zscore_shuffs": 500,
    "cv_folds": 10,
    "cv_metric": "rmse",
    "cv_grid": {
        #"alpha": [.005, .01, .05],
        "fit_intercept": [True,False],
        #"random_state": [0]
    }
}


def _get_latest_pkl():
    return max(glob(models_dir + "/*.pkl"), key=os.path.getctime)


def single_neuron_correlations(cfg, kinvars=None):

    if kinvars is None:
        #kinvars = ["spd2", "spd0", "spd1", "velx", "vely", "acc2", "accx", "accy"]
        kinvars = ["spd2", "spd0", "spd1", "acc2"]

    seg_data = SegmentData(**cfg["data"])

    obj = {
        "cfg": cfg,
        "neuron_ixs": seg_data.neuron_ixs,
        "fits": {}
    }

    C = np.zeros((len(seg_data.neuron_ixs), len(kinvars)), float)
    P = np.zeros_like(C)
    for kinvar in kinvars:
        print("-- " + kinvar + ": ")
        x, y, _, _ = seg_data.get_data(kinvar)
        for j in range(x.shape[1]):
            res = stats.pearsonr(x[:, j], y)
            C[j, kinvars.index(kinvar)] = res.statistic
            P[j, kinvars.index(kinvar)] = res.pvalue

    max_pval = .05
    c = np.zeros((len(seg_data.partic_neurons), len(kinvars)))
    for neuron_ix in seg_data.partic_neurons:
        c_ = np.abs(C[seg_data.neuron_ixs == neuron_ix, :])
        p_ = P[seg_data.neuron_ixs == neuron_ix, :]
        c_[p_ > max_pval] = 0
        c[neuron_ix, :] = np.max(c_, axis=0)

    return c


if __name__ == "__main__":
    #single_neuron_correlations(default_cfg)
    cv_fit_models(default_cfg, dump=True)
    pkl = _get_latest_pkl()
    print("Analyzing " + pkl)
    analyze_models(pkl)
