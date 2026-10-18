"""LaTeX-typeset animations for the NCVD manual and slide deck.

    python make_animations.py [names...] [--frames N] [--still-only]

Writes <name>.mp4 (H.264, 1600x700, 30 fps) and <name>_still.pdf (vector
still used in the manual). Uses the toolbox itself for every computed quantity.
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python'))
import ncvd as nc
from ncvd import systems as sy, plotting as pl

INK, PAPER, REF = '#132433', '#F7F6F2', '#8A8578'
C = pl.PALETTES['nld']
pl.set_style('nld', size=13)
plt.rcParams.update({'figure.facecolor': PAPER, 'savefig.facecolor': PAPER, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'axes.edgecolor': INK, 'savefig.bbox': None})
FIG = (8, 3.5); DPI = 200


def save(fig, anim, name, nframes, still_frame, still_only):
    anim._func(still_frame)
    fig.savefig(os.path.join(HERE, name + '_still.pdf'))
    if still_only:
        plt.close(fig); return
    w = FFMpegWriter(fps=30, codec='libx264', bitrate=4000,
                     extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'main', '-level:v', '4.0', '-g', '30',
                                 '-tag:v', 'avc1', '-colorspace', 'bt709', '-color_primaries', 'bt709',
                                 '-color_trc', 'bt709', '-brand', 'mp42', '-movflags', '+faststart'])
    anim.save(os.path.join(HERE, name + '.mp4'), writer=w, dpi=DPI)
    plt.close(fig)
    print('wrote', name)


# ------------------------------------------------------------------ 1 complex mode
def complex_mode(nf, still_only):
    """A complex mode is a rotating, shrinking polygon; motion is its real shadow."""
    M, Cm, K = sy.chain(6, 1, 4, 0.02, {0: 1.0, 1: 1.0})   # dampers at masses 1, 2 only
    lam, E = sy.true_modes(M, Cm, K)
    i = 3
    psi = nc.normalize_shape(E[:, i]); l = lam[i]
    nst = len(psi)
    _, mp = nc.realify(psi)
    T = 2*np.pi/abs(l.imag)*2.2
    ts = np.linspace(0, T, nf)
    fig = plt.figure(figsize=FIG, dpi=DPI)
    ax1 = fig.add_axes([0.085, 0.15, 0.34, 0.78]); ax2 = fig.add_axes([0.54, 0.15, 0.43, 0.75])
    R = 1.1*np.abs(psi).max()
    th = np.linspace(0, 2*np.pi, 200)
    for ax in [ax1]:
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect('equal')
        ax.axhline(0, color=REF, lw=0.6); ax.axvline(0, color=REF, lw=0.6)
        ax.set_xlabel(r'$\mathrm{Re}$'); ax.set_ylabel(r'$\mathrm{Im}$')
    poly, = ax1.plot([], [], '-o', color=C[0], ms=4, lw=1.4)
    drops = [ax1.plot([], [], ':', color=C[1], lw=0.8)[0] for _ in range(nst)]
    shadow, = ax1.plot([], [], 'o', color=C[1], ms=4)
    ax1.set_title(r'$\psi_j\,e^{\lambda t}$ in the complex plane', fontsize=12)
    x = np.arange(1, nst + 1)
    ax2.set_xlim(0.5, nst + 0.5); ax2.set_ylim(-R*1.25, R*1.25)
    ax2.axhline(0, color=REF, lw=0.6)
    ax2.set_xlabel(r'station $j$'); ax2.set_ylabel(r'$q_j(t)=\mathrm{Re}\,\psi_j e^{\lambda t}$')
    ghost = [ax2.plot([], [], '-', color=REF, lw=0.5, alpha=0.5)[0] for _ in range(6)]
    bars, = ax2.plot([], [], '-o', color=C[1], ms=5, lw=1.4)
    ax2.set_title(rf'6-DOF chain, mode {i+1}: $\lambda={l.real:.3f}+{l.imag:.3f}\,\mathrm{{i}}$, MPC $={mp:.2f}$', fontsize=11)
    ax2.text(0.02, 0.04, r'no common phase: the nodes travel', transform=ax2.transAxes, fontsize=10, color=REF)
    hist = []

    def f(k):
        z = psi*np.exp(l*ts[k])
        poly.set_data(np.r_[0, z.real], np.r_[0, z.imag])
        for j in range(nst):
            drops[j].set_data([z.real[j], z.real[j]], [z.imag[j], 0])
        shadow.set_data(z.real, 0*z.real)
        bars.set_data(x, z.real)
        if k % 6 == 0:
            hist.append(z.real.copy())
        for g, h in zip(ghost, hist[-6:][::-1]):
            g.set_data(x, h)
        return [poly, shadow, bars, *drops, *ghost]
    anim = FuncAnimation(fig, f, frames=nf, blit=False)
    save(fig, anim, 'complex_mode', nf, nf//7, still_only)


# ------------------------------------------------------------------ 2 Joukowski fold
def joukowski(nf, still_only):
    """Balanced quotient maps lam -> mu = (lam/c + c/lam)/2; poles with |lam/c| = 1
    meet their conjugates. Right panel: the off-diagonal ratio of Sigma_2."""
    n = 4
    K = 2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1); K[-1, -1] = 1
    Cd = 0.02*np.eye(n) + 0.01*K
    lam = np.linalg.eigvals(sy.state_matrix(np.eye(n), Cd, K))
    w = np.sort(np.abs(lam))[::2]
    cs = np.r_[np.geomspace(0.25, 2.6, nf//2), np.geomspace(2.6, 0.25, nf - nf//2)]
    fig = plt.figure(figsize=FIG, dpi=DPI)
    ax1 = fig.add_axes([0.05, 0.15, 0.27, 0.75]); ax2 = fig.add_axes([0.37, 0.15, 0.27, 0.75])
    ax3 = fig.add_axes([0.72, 0.17, 0.26, 0.73])
    th = np.linspace(0, 2*np.pi, 300)
    ax1.plot(np.cos(th), np.sin(th), '--', color=REF, lw=0.8)
    ax1.set_xlim(-2.8, 1.2); ax1.set_ylim(-2.8, 2.8); ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel(r'$\mathrm{Re}\,\lambda/c$'); ax1.set_ylabel(r'$\mathrm{Im}\,\lambda/c$')
    ax1.set_title(r'poles, time unit $1/c$', fontsize=11)
    ax2.plot([-1, 1], [0, 0], '-', color=REF, lw=2.5, alpha=0.5)
    ax2.set_xlim(-1.6, 1.6); ax2.set_ylim(-1.6, 1.6); ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel(r'$\mathrm{Re}\,\mu$'); ax2.set_ylabel(r'$\mathrm{Im}\,\mu$')
    ax2.set_title(r'$\mu=\frac{1}{2}\left(\lambda/c+c/\lambda\right)$', fontsize=11)
    order = np.argsort(np.abs(lam))
    pts1 = [ax1.plot([], [], 'o', color=C[k//2], ms=6, mfc=('none' if k % 2 else C[k//2]))[0] for k in range(8)]
    pts2 = [ax2.plot([], [], 'o', color=C[k//2], ms=6, mfc=('none' if k % 2 else C[k//2]))[0] for k in range(8)]
    # off-diagonal of the balanced basis on an exact record, precomputed
    t = np.arange(0, 120, 0.05)
    d, v, a = sy.simulate_linear(np.eye(n), Cd, K, np.random.default_rng(1).standard_normal(2*n), t)
    grid = np.unique(np.r_[np.geomspace(0.25, 2.6, 400), np.concatenate([x*np.geomspace(0.999, 1.001, 41) for x in w]), w])

    def offd(c):
        r = nc.ncvd(d, v/c, a/c**2)
        S = r.S2; D = np.diag(np.diag(S)); return np.linalg.norm(S - D)/np.linalg.norm(D)
    od = np.array([offd(c) for c in grid])
    gap = np.array([min(abs(0.5*(x/c + c/x) - 0.5*(y/c + c/y)) for i, x in enumerate(lam) for j, y in enumerate(lam) if i < j) for c in grid])
    ax3.loglog(grid, gap, '-', color=C[0], lw=1.2)
    ax3.loglog(grid, np.maximum(od, 1e-16), '-', color=C[1], lw=1.0)
    # direct labels in empty regions instead of a legend (a legend hides the curves)
    ax3.text(0.38, 2e-5, 'smallest\ngap of $\mu$', color=C[0], fontsize=9, linespacing=1.1)
    ax3.text(0.38, 1e-11, 'off-diagonal\nof $\Sigma_2$', color=C[1], fontsize=9, linespacing=1.1)
    for x in w:
        ax3.axvline(x, color=REF, ls=':', lw=0.7)
    ax3.set_xlabel(r'$c$'); ax3.set_ylim(1e-16, 10)
    from matplotlib.ticker import NullFormatter, FixedLocator
    ax3.xaxis.set_major_locator(FixedLocator([0.3, 1, 2])); ax3.xaxis.set_minor_formatter(NullFormatter())
    ax3.set_xticklabels([r'$0.3$', r'$1$', r'$2$'])
    ax3.set_yticks([1e-15, 1e-10, 1e-5, 1])
    ax3.set_title(r'ties at $c=|\lambda_k|$', fontsize=11)
    cur = ax3.axvline(1, color=INK, lw=1.0)
    lab = ax1.text(0.03, 0.03, '', transform=ax1.transAxes, fontsize=10)

    def f(k):
        c = cs[k]
        for j, idx in enumerate(np.r_[[o for o in order if lam[o].imag > 0], [o for o in order if lam[o].imag < 0]]):
            pass
        pos = sorted([i for i in range(8) if lam[i].imag > 0], key=lambda i: abs(lam[i]))
        neg = [int(np.argmin(np.abs(lam - lam[i].conjugate()))) for i in pos]
        seq = [x for pair in zip(pos, neg) for x in pair]
        for p1, p2, i in zip(pts1, pts2, seq):
            z = lam[i]/c; mu = 0.5*(z + 1/z)
            p1.set_data([z.real], [z.imag]); p2.set_data([mu.real], [mu.imag])
        cur.set_xdata([c, c])
        lab.set_text(rf'$c={c:.2f}$')
        return pts1 + pts2 + [cur, lab]
    anim = FuncAnimation(fig, f, frames=nf, blit=False)
    save(fig, anim, 'joukowski_fold', nf, int(np.argmin(np.abs(cs[:nf//2] - w[1]))), still_only)


# ------------------------------------------------------------------ 3 residual geometry
def residual(nf, still_only):
    """X2 = U Lam Phi^H + R: the fit is the projection of each rate column onto
    span(X1); R is orthogonal to it. Shown for one rate column in 3-D."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=FIG, dpi=DPI)
    ax = fig.add_axes([-0.05, -0.12, 0.66, 1.25], projection='3d')
    ax.set_facecolor(PAPER)
    s = np.linspace(-1, 1, 2)
    P, Qm = np.meshgrid(s*1.3, s*1.3)
    ax.plot_surface(P, Qm, 0*P, color=C[0], alpha=0.12, linewidth=0)
    ax.quiver(0, 0, 0, 1, 0, 0, color=C[0], lw=2, arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, 0.3, 0.9, 0, color=C[0], lw=2, arrow_length_ratio=0.08)
    y = np.array([0.8, 0.6, 0.55])
    ax.quiver(0, 0, 0, *y, color=C[1], lw=2.4, arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, y[0], y[1], 0, color=C[2], lw=2.4, arrow_length_ratio=0.08)
    ax.plot([y[0], y[0]], [y[1], y[1]], [0, y[2]], '--', color=C[4], lw=2)
    ax.text(1.05, -0.05, 0, r'$\mathbf{x}_1$-columns', color=C[0], fontsize=10)
    ax.text(*(y + [0.02, 0.02, 0.06]), r'$\dot{\mathbf{x}}$ (a rate column)', color=C[1], fontsize=10)
    ax.text(y[0] + 0.05, y[1], 0.02, r'$\mathbf{U}\boldsymbol{\Lambda}\boldsymbol{\Phi}^{\mathsf H}$', color=C[2], fontsize=11)
    ax.text(y[0] + 0.05, y[1] + 0.02, y[2]/2, r'$\mathbf{R}$', color=C[4], fontsize=12)
    ax.set_xlim(-0.2, 1.2); ax.set_ylim(-0.2, 1.2); ax.set_zlim(0, 0.9)
    ax.set_axis_off()
    tx = fig.add_axes([0.60, 0.08, 0.39, 0.84]); tx.axis('off')
    tx.text(0, 0.95, r'$\mathbf{X}_1=\mathbf{U}\boldsymbol{\Phi}^{\mathsf H}$ \quad exact', fontsize=14, va='top')
    tx.text(0, 0.80, r'$\mathbf{X}_2=\mathbf{U}\boldsymbol{\Lambda}\boldsymbol{\Phi}^{\mathsf H}+\mathbf{R}$', fontsize=14, va='top')
    tx.text(0, 0.62, r'$\mathbf{R}=(\mathbf{I}-\mathbf{X}_1\mathbf{X}_1^{\dagger})\mathbf{X}_2\ \perp\ \mathrm{span}\,\mathbf{X}_1$',
            fontsize=13, va='top')
    tx.text(0, 0.42, r'linear, noise-free: $\mathbf{R}=\mathbf{0}$', fontsize=12, va='top', color=C[2])
    tx.text(0, 0.30, r'nonlinear window: $\|\mathbf{R}\|/\|\mathbf{X}_2\|$', fontsize=12, va='top', color=C[4])
    tx.text(0, 0.20, r'is what no linear model of the window explains', fontsize=11, va='top', color=REF)
    az = np.r_[np.linspace(-60, 30, nf//2), np.linspace(30, -60, nf - nf//2)]

    def f(k):
        ax.view_init(elev=22 + 8*np.sin(2*np.pi*k/nf), azim=az[k])
        return []
    anim = FuncAnimation(fig, f, frames=nf, blit=False)
    save(fig, anim, 'residual_projection', nf, nf//3, still_only)


# ------------------------------------------------------------------ 4 FIR band
def fir_band(nf, still_only):
    """A derivative stencil sees exp(i w dt) through its symbol; the inversion is
    unique only while Im d(exp(i w dt)) increases."""
    names = ['forward', 'central', 'central4', 'savgol']
    labels = ['forward', 'central', 'five-point', r'Savitzky--Golay (7,3)']
    fl = [nc.deriv_filter(m) for m in names]
    wg = np.linspace(1e-3, np.pi, 600)
    fig = plt.figure(figsize=FIG, dpi=DPI)
    ax1 = fig.add_axes([0.05, 0.15, 0.3, 0.75]); ax2 = fig.add_axes([0.45, 0.15, 0.52, 0.75])
    th = np.linspace(0, 2*np.pi, 300)
    ax1.plot(np.cos(th), np.sin(th), '-', color=REF, lw=0.8)
    ax1.set_aspect('equal'); ax1.set_xlim(-1.3, 1.3); ax1.set_ylim(-1.3, 1.3)
    ax1.set_xlabel(r'$\mathrm{Re}\,\theta$'); ax1.set_ylabel(r'$\mathrm{Im}\,\theta$')
    ax1.set_title(r'$\theta=e^{\mathrm{i}\omega\Delta t}$', fontsize=12)
    pt, = ax1.plot([], [], 'o', color=INK, ms=7)
    arc, = ax1.plot([], [], '-', color=INK, lw=2)
    ax2.plot(wg, wg, '--', color=REF, lw=1, label=r'exact, $\omega\Delta t$')
    dots = []
    for k, (f_, lb) in enumerate(zip(fl, labels)):
        g = np.imag(f_.symbol(np.exp(1j*wg), 1.0))
        if names[k] == 'forward':
            g = np.abs(f_.symbol(np.exp(1j*wg), 1.0))
        ax2.plot(wg, g, '-', color=C[k], lw=1.5, label=lb)
        b = f_.band()
        if b < np.pi - 1e-6:
            ax2.axvline(b, color=C[k], ls=':', lw=1)
        dots.append((ax2.plot([], [], 'o', color=C[k], ms=6)[0], f_, names[k] == 'forward'))
    ax2.set_xlim(0, np.pi); ax2.set_ylim(0, 2.2)
    ax2.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax2.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    ax2.set_xlabel(r'$\omega\Delta t$'); ax2.set_ylabel(r'$|d(\theta)|\,\Delta t$ (forward), $\mathrm{Im}\,d(\theta)\,\Delta t$')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.set_title(r'the correction is unique while the curve rises (dotted: end of band)', fontsize=11)
    ws = np.r_[np.linspace(0.02, np.pi - 0.02, nf)]

    def f(k):
        w = ws[k]
        pt.set_data([np.cos(w)], [np.sin(w)])
        a = np.linspace(0, w, 60); arc.set_data(0.35*np.cos(a), 0.35*np.sin(a))
        for d, f_, fw in dots:
            s = f_.symbol(np.exp(1j*w), 1.0)
            d.set_data([w], [abs(s) if fw else s.imag])
        return [pt, arc] + [d for d, _, _ in dots]
    anim = FuncAnimation(fig, f, frames=nf, blit=False)
    save(fig, anim, 'fir_band', nf, int(0.42*nf), still_only)


# ------------------------------------------------------------------ 5 backbone tracking
def backbone(nf, still_only):
    """Sliding-window NCVD along a Duffing decay traces frequency-energy dependence."""
    M, _, K, fnl = sy.duffing_chain(3, k=1.0, c=0.0, k3={0: 1.0}); Cd = 0.003*M
    lam0, E0 = sy.true_modes(M, Cd, K)
    sh = [nc.realify(E0[:, i])[0] for i in range(3)]; sh = [x/np.abs(x).max() for x in sh]
    x0 = np.r_[1.5*sh[0] + 0.6*sh[1] + 0.5*sh[2], np.zeros(3)]
    fs = 20.0; t = np.arange(0, 1500, 1/fs)
    d, v, a = sy.simulate_nonlinear(M, Cd, K, fnl, x0, t)
    L, S = 800, 200
    tr = nc.track(d, v, a, fs=fs, length=L, step=S, n_track=3, M=M)
    nw = len(tr.t)
    fig = plt.figure(figsize=FIG, dpi=DPI)
    ax1 = fig.add_axes([0.07, 0.66, 0.9, 0.29]); ax2 = fig.add_axes([0.07, 0.17, 0.22, 0.33])
    ax3 = fig.add_axes([0.40, 0.17, 0.25, 0.33]); ax4 = fig.add_axes([0.74, 0.17, 0.23, 0.33])
    ax1.plot(t, d[:, 0], '-', color=C[0], lw=0.4)
    ax1.set_xlim(0, t[-1]); ax1.set_ylabel(r'$q_1$')
    ax1.text(0.995, 0.92, r'$t$ (s)', transform=ax1.transAxes, ha='right', va='top', fontsize=11)
    ax1.set_title('one free decay; the shaded window is decomposed', fontsize=11)
    span = ax1.axvspan(0, L/fs, color=C[1], alpha=0.25)
    ax2.set_xlabel(r'$q_1$', labelpad=1); ax2.set_ylabel(r'$\dot q_1$', labelpad=1)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_title('window orbit', fontsize=10)
    lim = 1.05*np.abs(d[:, 0]).max(); vl = 1.05*np.abs(v[:, 0]).max()
    ax2.set_xlim(-lim, lim); ax2.set_ylim(-vl, vl)
    orb, = ax2.plot([], [], '-', color=C[0], lw=0.6)
    ax3.set_xscale('log'); ax3.set_xlabel('modal energy'); ax3.set_ylabel(r'$\omega_d/\omega_{d,\mathrm{lin}}$')
    ax3.axhline(1, color=REF, ls='--', lw=0.8)
    E = tr.energy; rat = tr.wd/np.abs(lam0.imag)[None, :]
    ax3.set_xlim(np.nanmin(E)*0.7, np.nanmax(E)*1.5); ax3.set_ylim(0.98, np.nanmax(rat)*1.02)
    bb = [ax3.plot([], [], '-o', color=C[j], ms=2.5, lw=1, label=rf'mode {j+1}')[0] for j in range(3)]
    ax3.plot([], [], '--', color=REF, lw=0.8, label='linear')
    ax3.legend(fontsize=8, loc='upper left', handlelength=1.5, borderaxespad=0.3)
    ax4.set_yscale('log'); ax4.set_xlim(0, t[-1]); ax4.set_ylim(np.nanmin(tr.res)*0.5, np.nanmax(tr.res)*2)
    ax4.set_xlabel(r'$t$ (s)'); ax4.set_ylabel(r'$\|\mathbf{R}\|/\|\mathbf{X}_2\|$')
    rr, = ax4.plot([], [], '-o', color=C[4], ms=2.5, lw=1, label='window residual')
    ax4.legend(fontsize=8, loc='upper right', borderaxespad=0.3)
    frames = np.linspace(0, nw - 1, nf).astype(int)

    def f(k):
        w = frames[k]
        a0, b0 = tr.edges[w]
        span.set_x(a0/fs); span.set_width((b0 - a0)/fs)
        orb.set_data(d[a0:b0, 0], v[a0:b0, 0])
        ql = 1.1*np.abs(d[a0:b0, 0]).max(); vq = 1.1*np.abs(v[a0:b0, 0]).max()
        ax2.set_xlim(-ql, ql); ax2.set_ylim(-vq, vq)
        for j in range(3):
            bb[j].set_data(E[:w+1, j], rat[:w+1, j])
        rr.set_data(tr.t[:w+1], tr.res[:w+1])
        return [span, orb, rr, *bb]
    anim = FuncAnimation(fig, f, frames=nf, blit=False)
    save(fig, anim, 'backbone_tracking', nf, nf - 1, still_only)


ALL = dict(complex_mode=complex_mode, joukowski_fold=joukowski, residual_projection=residual,
           fir_band=fir_band, backbone_tracking=backbone)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('names', nargs='*')
    p.add_argument('--frames', type=int, default=240)
    p.add_argument('--still-only', action='store_true')
    a = p.parse_args()
    for nm in (a.names or ALL):
        ALL[nm](a.frames, a.still_only)
