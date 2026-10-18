"""Ensemble NCVD of several records with different lengths, sampling rates or noise.

Because the state-rate pair is a continuous-time relation, records sampled at
different rates share the same operator, X2 = X1 A^T. Stacking their rows and
decomposing once is the least-squares fusion of all records; it replaces the
quotient averaging of Eq. (59). Each record keeps its own block of the
temporal basis U.
"""
import numpy as np
from .core import ncvd_pair

__all__ = ['ncvd_ensemble']


def ncvd_ensemble(records, weights='equal', center=False, **kw):
    """Joint NCVD of a list of records.

    records : list of (disp, vel, acc) tuples (each (m_i, n), same n).
    weights : 'equal'   every record contributes equally (row weight 1/m_i),
              'samples' every sample contributes equally (weight 1),
              'snr'     weight 1/(m_i * noise_var_i), noise_var_i estimated from
                        the regression residual of each record alone,
              or an array of per-record weights.
    center : remove each record's own mean before stacking.
    Other keyword arguments go to `ncvd_pair` (rank, operator, time_scale, ...).

    Returns the NCVDResult of the stacked record plus
        'blocks' : list of slices giving each record's rows in U,
        'record_weights' : the weights used.
    """
    Y1s, Y2s, ws, blocks = [], [], [], []
    row = 0
    nst = None
    for i, rec in enumerate(records):
        d, v, a = (np.asarray(x, float) for x in rec)
        nst = d.shape[1] if nst is None else nst
        if d.shape[1] != nst:
            raise ValueError('all records must have the same number of stations')
        Y1 = np.hstack([d, v]); Y2 = np.hstack([v, a])
        if center:
            Y1 = Y1 - Y1.mean(0); Y2 = Y2 - Y2.mean(0)
        Y1s.append(Y1); Y2s.append(Y2)
        blocks.append(slice(row, row + len(Y1))); row += len(Y1)
    m = np.array([len(y) for y in Y1s], float)
    if isinstance(weights, str):
        if weights == 'equal':
            w = 1.0/m
        elif weights == 'samples':
            w = np.ones(len(m))
        elif weights == 'snr':
            nv = []
            for Y1, Y2 in zip(Y1s, Y2s):
                B = np.linalg.lstsq(Y1, Y2, rcond=None)[0]
                nv.append(np.mean((Y2 - Y1 @ B)**2)/max(np.mean(Y2**2), 1e-300))
            w = 1.0/(m*np.maximum(np.array(nv), 1e-16))
        else:
            raise ValueError(weights)
    else:
        w = np.asarray(weights, float)
    w = w/w.max()
    rw = np.concatenate([np.full(int(mi), wi) for mi, wi in zip(m, w)])
    r = ncvd_pair(np.vstack(Y1s), np.vstack(Y2s), weights=rw,
                  structured_split=nst, **kw)
    r['blocks'] = blocks
    r['record_weights'] = w
    return r
