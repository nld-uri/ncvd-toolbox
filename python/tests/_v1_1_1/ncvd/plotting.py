"""Figures with LaTeX text (Computer Modern) and a fixed categorical palette.

palette='matlab' uses MATLAB's default ColorOrder; palette='nld' uses the
Nonlinear Dynamics Laboratory figure series. Reference curves are dashed grey.
Save as vector PDF with `save(fig, 'name.pdf')`.
"""
import os
import shutil
import numpy as np

__all__ = ['set_style', 'PALETTES', 'plot_poles', 'plot_shapes', 'plot_backbone',
           'plot_tracking', 'save']

PALETTES = {
    'matlab': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F'],
    'nld': ['#1F5A85', '#B4622A', '#588A75', '#CA863E', '#774860'],
}
REF = {'matlab': '#808080', 'nld': '#8A8578'}
_STATE = {'palette': 'matlab'}


def set_style(palette='matlab', usetex=None, size=10):
    """LaTeX text and the chosen palette. usetex=None: use LaTeX if available."""
    import matplotlib as mpl
    if '/Library/TeX/texbin' not in os.environ.get('PATH', '') and os.path.isdir('/Library/TeX/texbin'):
        os.environ['PATH'] += os.pathsep + '/Library/TeX/texbin'
    if usetex is None:
        usetex = shutil.which('latex') is not None
    _STATE['palette'] = palette
    mpl.rcParams.update({
        'text.usetex': usetex,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'mathtext.fontset': 'cm',
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}',
        'axes.prop_cycle': mpl.cycler(color=PALETTES[palette]),
        'font.size': size, 'axes.labelsize': size, 'legend.fontsize': size - 1,
        'xtick.labelsize': size - 1, 'ytick.labelsize': size - 1,
        'axes.linewidth': 0.6, 'lines.linewidth': 1.3, 'figure.dpi': 120,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'axes.facecolor': 'white', 'figure.facecolor': 'white',
        'legend.frameon': False,
    })
    return usetex


def _c(i):
    p = PALETTES[_STATE['palette']]
    return p[i % len(p)]


def _ref():
    return REF[_STATE['palette']]


def plot_poles(ax, lam_list, labels=None, markers='os^vD', true=None):
    """Poles in the complex plane. lam_list: list of arrays."""
    if true is not None:
        ax.plot(np.real(true), np.imag(true), 'o', mfc='none', mec='k', ms=7, label='truth')
    for i, lam in enumerate(lam_list):
        ax.plot(np.real(lam), np.imag(lam), markers[i % len(markers)], color=_c(i), ms=4,
                mfc='none', label=None if labels is None else labels[i])
    ax.axvline(0, color=_ref(), lw=0.6, ls='--')
    ax.set_xlabel(r'$\mathrm{Re}\,\lambda$'); ax.set_ylabel(r'$\mathrm{Im}\,\lambda$')
    if labels is not None or true is not None:
        ax.legend()
    return ax


def plot_shapes(ax, shapes, labels=None, true=None):
    """Complex shapes as polygons in the complex plane (stations joined in order)."""
    if true is not None:
        t = np.r_[0, true]
        ax.plot(t.real, t.imag, '-', color=_ref(), lw=2.2, label='truth')
    for i, s in enumerate(np.atleast_2d(shapes)):
        z = np.r_[0, s]
        ax.plot(z.real, z.imag, '-o', color=_c(i), ms=3,
                label=None if labels is None else labels[i])
    ax.set_xlabel(r'$\mathrm{Re}\,\phi$'); ax.set_ylabel(r'$\mathrm{Im}\,\phi$')
    ax.set_aspect('equal', adjustable='datalim')
    return ax


def plot_backbone(ax, tr, x='amp', modes=None, hz=False, ref=None, labels=None):
    """Frequency against amplitude (or energy) for tracked resonances."""
    k = tr['wn'].shape[1]
    modes = range(k) if modes is None else modes
    for j in modes:
        y = tr['wd'][:, j]/(2*np.pi if hz else 1)
        ax.plot(tr[x][:, j], y, '-o', ms=3, color=_c(j),
                label=(labels[j] if labels else rf'mode {j+1}'))
        if ref is not None:
            ax.axhline(ref[j]/(2*np.pi if hz else 1), color=_ref(), ls='--', lw=0.8)
    ax.set_xlabel('modal amplitude' if x == 'amp' else 'modal energy')
    ax.set_ylabel(r'$\omega_d$ ' + ('(Hz)' if hz else '(rad/s)'))
    if x == 'energy':
        ax.set_xscale('log')
    ax.legend()
    return ax


def plot_tracking(tr, fig=None):
    """Four-panel summary: frequency, damping, MPC and residual against time."""
    import matplotlib.pyplot as plt
    fig = fig or plt.figure(figsize=(7, 4.6))
    axs = fig.subplots(2, 2, sharex=True)
    t = tr['t']
    for j in range(tr['wn'].shape[1]):
        axs[0, 0].plot(t, tr['wd'][:, j], '-o', ms=2.5, color=_c(j))
        axs[0, 1].plot(t, tr['zeta'][:, j], '-o', ms=2.5, color=_c(j))
        axs[1, 0].plot(t, tr['mpc'][:, j], '-o', ms=2.5, color=_c(j))
    axs[1, 1].semilogy(t, tr['res'], '-o', ms=2.5, color='k')
    axs[0, 0].set_ylabel(r'$\omega_d$'); axs[0, 1].set_ylabel(r'$\zeta$')
    axs[1, 0].set_ylabel('MPC'); axs[1, 1].set_ylabel(r'$\|\mathbf{R}\|/\|\mathbf{X}_2\|$')
    for a in axs[1]:
        a.set_xlabel(r'$t$')
    fig.tight_layout()
    return fig


def save(fig, path, dpi=200):
    """Save vector PDF (and raster if the extension is .png)."""
    fig.savefig(path, dpi=dpi)
    return path
