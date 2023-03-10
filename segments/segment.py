import numpy as np
from kinematics import KinData
from dataclasses import dataclass
import json
import os
from motorneural.data import DataSummary
import matplotlib.pyplot as plt
import tools
import path_mgr


@dataclass
class Segment:

    ix: int
    trial_ix: int
    slc: slice
    kin: KinData
    S: np.ndarray


def load_segments(json_file):
    obj = json.load(open(path_mgr.segments_dir() + os.path.sep + json_file, "r"))
    segs = []
    for s in obj["segs"]:
        segs.append(Segment(ix=s['ix'], trial_ix=s['trial_ix'], slc=slice(*s['slc']),
                            kin=KinData.from_json(s['kin']), S=np.array(s['S'])))
    del obj["segs"]

    data_summary = DataSummary(**obj['data_summary'])
    print(f"Loaded {len(segs)} segments from {len(set(s.trial_ix for s in segs))} trials, for data: {data_summary}")

    return segs, data_summary, obj['seg_cfg']


def show_segment(s: Segment, kins=None):
    if kins is None:
        kins = ["spd0", "spd1", "spd2", "crv2"]
    mosaic = [["X", "NEURAL"] + kins[:len(kins) // 2], ["X", "NEURAL"] + kins[len(kins) // 2:]]
    _, axs = plt.subplot_mosaic(mosaic)
    plt.gcf().set_size_inches(16, 8)
    for m in axs:
        if m == "X":
            axs[m].plot(s.kin[m][:, 0], s.kin[m][:, 1], "k.-")
            axs[m].axis("equal")
        elif m == "NEURAL":
            axs[m].imshow(s.S, cmap="hot")
        else:
            axs[m].plot(s.kin.t, s.kin[m], "g.-")
            axs[m].set_title(m)
    plt.show()



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
        y = tools.zscore(y[train_msk], y, axis=0, robust=True)

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
            self.x = tools.normalize(self.x, axis=0, kind=self.normalize_neural)

        if isinstance(self.partic_neurons, int) and self.partic_neurons == -1:
            self.partic_neurons = list(set(self.full_neuron_ixs))
        assert hasattr(self.partic_neurons, "__len__")

        np.random.set_state(rand_state)
