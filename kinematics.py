
import numpy as np
from motorneural.typechecking import *
from motorneural.motor import KinData
from scipy.interpolate import interp1d
from geometrik.spcurve_factory import make_ndspline
from geometrik.invariants import geometric_invariants


def calc_kinematics(X: NpNx2[float], t: NpVec[float], dst_t: NpVec[float], dx: float = .5):

    spl = make_ndspline(X=X, t=t, dx=dx, default_t=None, stol=.1)
    geom_invars = geometric_invariants(spl)
    vel = spl(der=1)
    acc = spl(der=2)

    kin = {'X': spl(),
           'velx': vel[:, 0],
           'vely': vel[:, 1],
           'accx': acc[:, 0],
           'accy': acc[:, 1],
           'spd2': np.linalg.norm(vel, axis=1),
           'acc2': np.linalg.norm(acc, axis=1),
           'spd1': np.gradient(geom_invars['s1'], spl.t, edge_order=1),
           'spd0': np.gradient(geom_invars['s0'], spl.t, edge_order=1),
           'crv2': geom_invars['k2']
           }

    fs = (len(dst_t) - 1) / (dst_t[-1] - dst_t[0])
    t0 = dst_t[0]
    deviation_from_uniform = np.max(np.abs(dst_t - (t0 + np.arange(len(dst_t)) / fs)))
    max_deviation = .01  # in dt units
    assert deviation_from_uniform * fs < max_deviation

    s = kin['spd2'].copy()
    assert np.all(kin['spd2'] >= 0)

    positives = ['spd0', 'spd1', 'spd2', 'acc2']

    for k, v in kin.items():
        vi = interp1d(spl.t, v, axis=0, kind="cubic")(dst_t)
        if k in positives:
            assert np.mean(vi < 0) < .5, (k, np.mean(vi < 0))
            vi[vi < 0] = 0
        kin[k] = vi

    return KinData(fs, t0, kin)


