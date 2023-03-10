import os
from motorneural.datasets.hatsopoulos import HatsoData
from motorneural.data import DataSummary
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, median_filter
import json
from kinematics import calc_kinematics

from segments.segment import Segment, segments_dir, show_segment


def dump_segments(data_summary: DataSummary, seg_cfg: dict, segs: list[Segment], json_file: str = None):

    if not json_file:
        json_file = os.path.join(segments_dir,
                                 str(data_summary) + f" dur{int(seg_cfg['seg_dur_sec'] * 1000)} - segments.json")
        os.makedirs(segments_dir, exist_ok=True)

    obj = {
        "data_summary": data_summary.__dict__,
        "seg_cfg": seg_cfg,
        "segs": []
    }

    for s in segs:
        obj["segs"].append({
            "ix": s.ix,
            "trial_ix": s.trial_ix,
            "slc": [int(s.slc.start), int(s.slc.stop)],
            "kin": s.kin.to_json(),
            "S": s.S.tolist()
        })

    json.dump(obj, open(json_file, "w"), indent=4)
    print(f"Dumped {len(segs)} segments from {len(set(s.trial_ix for s in segs))} trials, to: " + json_file)
    return json_file


def collect_segments(data, seg_cfg: dict, rand_seed=0, dump=True):

    min_crv = seg_cfg["min_crv"]
    max_crv = seg_cfg["max_crv"]
    smooth_dur = seg_cfg["smooth_dur"]
    seg_dur_sec = seg_cfg["seg_dur_sec"]

    seg_dur_ix = int(.5 + seg_dur_sec / data.bin_sz)

    segs = []
    for tr in data:
        np.random.seed(rand_seed)

        crv = np.abs(tr.kin['crv2'])
        crv = median_filter(crv, int(.5 + smooth_dur / tr.bin_sz), mode="mirror")
        crv = gaussian_filter1d(crv, sigma=smooth_dur / tr.bin_sz, mode="mirror")

        all_peaks_mask = np.zeros(tr.num_samples, bool)
        all_peaks_mask[find_peaks(crv, min_crv)[0]] = True
        valid_peak_ixs = find_peaks(crv, [min_crv, max_crv])[0]

        is_segment = np.zeros(tr.num_samples, bool)

        for ix in np.random.permutation(valid_peak_ixs):
            start = ix - seg_dur_ix // 2
            slc = slice(start, start + seg_dur_ix)
            if slc.start < 0 or slc.stop > tr.num_samples or np.any(is_segment[slc]) or np.sum(all_peaks_mask[slc]) > 1:
                continue
            is_segment[slc] = True
            segs.append(Segment(ix=len(segs) + 1, trial_ix=tr.ix, slc=slc,
                                kin=tr.kin.get_slice(slc), S=tr.neural.spkcounts[:, slc]))

        _show_segs = False
        _dbg_plot = False

        if _show_segs:
            for seg in segs:
                show_segment(seg)

        if _dbg_plot:

            mark_ixs = valid_peak_ixs

            color = np.log(.001 + crv)
            X = tr.kin['X']

            # ----

            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], c=color, cmap="hot", s=5)
            for seg in segs:
                plt.plot(seg.kin.X[:, 0], seg.kin.X[:, 1], '-')
            for ix in mark_ixs:
                plt.plot(X[ix, 0], X[ix, 1], 'sk')
                plt.text(X[ix, 0], X[ix, 1], str(ix), color='r')
            plt.axis("equal")
            plt.title(f"TRAJ - {tr.ix}")

            # ----

            use_rad_curv = True

            plt.figure()
            if use_rad_curv:
                crv = 1 / crv
            p = np.percentile(crv, 90)
            crv[crv > p] = p
            plt.plot(np.arange(len(crv)), crv, 'k-')
            plt.scatter(np.arange(len(crv)), crv, c=color, cmap="hot", s=5)
            for ix in mark_ixs:
                plt.plot(ix, crv[ix], 'sk')
                plt.text(ix, crv[ix], str(ix), color='r')
            plt.title("RADIUS OF CURVATURE" if use_rad_curv else "CURVATURE")

            # ----

            plt.show()

    if dump:
        dump_segments(data.summary, seg_cfg, segs)
    return segs


if __name__ == "__main__":
    seg_cfg = {
        "min_crv": 1 / 50,
        "max_crv": 1 / .5,
        "smooth_dur": .05,
        "seg_dur_sec": .1
    }
    for dataset in ("TP_RJ", "TP_RS"):
        print("Loading data..")
        data = HatsoData.make("~/data/hatsopoulos", dataset, lag=.1, bin_sz=.01, kin_fnc=calc_kinematics)
        print("Making segments..")
        collect_segments(data, seg_cfg)
