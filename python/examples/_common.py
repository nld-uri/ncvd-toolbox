"""Shared CLI for the examples: --out DIR (figure folder), --palette matlab|nld."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def setup(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'))
    p.add_argument('--palette', default='matlab', choices=['matlab', 'nld'])
    p.add_argument('--quick', action='store_true', help='fewer Monte Carlo trials')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from ncvd import plotting
    plotting.set_style(a.palette)
    return a


def fin(fig, out, name):
    import matplotlib.pyplot as plt
    path = os.path.join(out, name)
    fig.savefig(path)
    plt.close(fig)
    print('wrote', path)
