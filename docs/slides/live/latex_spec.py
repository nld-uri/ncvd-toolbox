import sys
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), 'lib'))
from latexblock import items
B = []
def add(*a): B.append(a)
add('identities', 'ncvd_identities', 1423, 44, 1.5, items([
    r'$\symbf{U}$: temporal basis. $\symbf{\Phi}$: characteristic shapes. $\lambda_i$: characteristic values (poles).',
    r'$\symbf{R}$: what the diagonal form cannot represent: noise, unsupplied forcing, nonlinearity.',
    r'Linear, noise-free record: $\symbf{R}=\symbf{0}$ to machine precision, and $\lambda_i$ belongs to column $i$ of $\symbf{\Phi}$ by construction.']),
    'U: temporal basis. Phi: characteristic shapes. lambda_i: characteristic values (poles). R: what the diagonal form cannot represent. Linear, noise-free record: R = 0 to machine precision, and lambda_i belongs to column i of Phi by construction.')
add('published', 'ncvd_published', 1515, 44, 1.5, items([
    r'Exact only when $\symbf{V}$ diagonalises $\symbf{\Sigma}_2$, that is, when $\symbf{V}$ are eigenvectors of $\symbf{B}^{\mathsf{H}}$.',
    r'True for the CVD and “forced” quotients on any data, and for the balanced quotient on exact linear data.',
    r'Not true for averaged ensemble quotients, nonlinear windows, or exact Joukowski ties.']),
    'Exact only when V diagonalises Sigma_2, that is, when V are eigenvectors of B^H. True for the CVD and forced quotients on any data, and for the balanced quotient on exact linear data. Not true for averaged ensemble quotients, nonlinear windows, or exact Joukowski ties.')
add('bcls', 'ncvd_bcls', 560, 36, 1.4,
    r'$\symbf{N}$: white sensor-noise variances, estimated from the flat spectrum near Nyquist.\par\medskip At 15\,dB: frequency error 10\,\% (LS) against 1.9\,\% (BCLS).\par\medskip TLS fails under unmeasured forcing.',
    'N: white sensor-noise variances, estimated from the flat spectrum near Nyquist. At 15 dB: frequency error 10% (LS) against 1.9% (BCLS). TLS fails under unmeasured forcing.')
add('fir', 'ncvd_fir', 444, 36, 1.4,
    r'Machine precision for forward, central, five-point and Savitzky–Golay rates.\par\medskip Invertible bands: $\omega\Delta t<\pi,\ \pi/2,\ 1.80,\ 1.14$.',
    'Machine precision for forward, central, five-point and Savitzky–Golay rates. Invertible bands: omega dt < pi, pi/2, 1.80, 1.14.')
add('readout-error', 'ncvd_readout_cap', 1500, 28, 1.35,
    r'\color{muted}Left: exact tie at $\omega=1$. Right: windows of a Duffing decay; fitted slopes 1.74 (poles) and 1.05 (shapes).',
    'Left: exact tie at omega = 1. Right: windows of a Duffing decay; fitted slopes 1.74 (poles) and 1.05 (shapes).')
