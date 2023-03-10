import os
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from segments.segment import SegmentData
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import KFold, ParameterGrid
from glob import glob
import neuralpop
import tools
import path_mgr


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
            zscores[mtrc].append(tools.zscore(xs=vals[1:], x=vals[0], robust=False))

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
        pkl = path_mgr.models_dir() + f"/{run_name}.pkl"
        pickle.dump(obj, open(pkl, 'wb'))
        print("Dumped: " + pkl)

    return obj




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
    plt.xlim(xticks[0], xticks[-1])
    plt.xticks(ticks=xticks, labels=xticks)
    plt.legend(loc='upper right')
    plt.xlabel('R2 zscore')


def analyze_models(pkl):
    obj = pickle.load(open(pkl, 'rb'))

    print("-- Calculating populations")
    populations = {}
    for kinvar in obj['fits']:
        print(" " + kinvar + ": ", end="")
        populations[kinvar] = neuralpop.calc_neural_populations(obj['fits'][kinvar]['model'], obj['neuron_ixs'],
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
    return max(glob(path_mgr.models_dir() + "/*.pkl"), key=os.path.getctime)


if __name__ == "__main__":
    #single_neuron_correlations(default_cfg)
    cv_fit_models(default_cfg, dump=True)
    pkl = _get_latest_pkl()
    print("Analyzing " + pkl)
    analyze_models(pkl)
