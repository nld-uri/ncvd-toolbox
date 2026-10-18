"""Build the WordPress block markup of web.uri.edu/nld/ncvd-toolbox/ (page.html).

Mirrors the kinematics-toolbox page: navy band, intro, buttons, eyebrow labels
with claim headings, MathML equations generated from LaTeX, figures served from
this repository (raw.githubusercontent.com).
"""
import os, html
from latex2mathml.converter import convert

RAW = 'https://raw.githubusercontent.com/nld-uri/ncvd-toolbox/main/docs/web/'
GH = 'https://github.com/nld-uri/ncvd-toolbox'
MSTYLE = "font-family:'Latin Modern Math','STIX Two Math','Cambria Math',math;font-size:1.05em"
B = []


def blk(kind, inner, attrs=''):
    a = f' {attrs}' if attrs else ''
    B.append(f'<!-- wp:{kind}{a} -->\n{inner}\n<!-- /wp:{kind} -->')


def m(tex, display=False):
    x = convert(tex, display='block' if display else 'inline')
    body = x[x.index('>') + 1:x.rindex('</math>')]
    ann = f'<annotation encoding="application/x-tex">{html.escape(tex, quote=False)}</annotation>'
    if display:
        return (f'<math display="block" style="display:block math;{MSTYLE.replace("1.05em", "1.15em")};margin:16px 0">'
                f'<semantics>{body}{ann}</semantics></math>')
    return f'<math style="{MSTYLE}"><semantics>{body}{ann}</semantics></math>'


def p(text):
    blk('paragraph', f'<p>{text}</p>')


def eyebrow(t):
    blk('html', '<p style="margin:40px 0 8px;font-size:13px;line-height:16px;letter-spacing:0.12em;text-transform:uppercase;'
        'font-weight:600;color:#5B6168"><span style="display:inline-block;width:24px;height:4px;background:#B4622A;'
        f'vertical-align:middle;margin-right:10px"></span>{t}</p>')


def h2(t):
    blk('heading', f'<h2 class="wp-block-heading">{t}</h2>')


def fig(file, alt, cap):
    blk('html', f'<figure style="margin:24px 0"><img src="{RAW}{file}" alt="{alt}" style="width:100%;height:auto;'
        f'border:1px solid #DCD9D2"><figcaption style="font-size:14px;color:#5B6168;margin-top:8px">{cap}</figcaption></figure>')


def code(src):
    blk('code', f'<pre class="wp-block-code"><code>{html.escape(src)}</code></pre>')


X1, X2 = r'\mathbf{X}_1', r'\mathbf{X}_2'
# ---------------------------------------------------------------- header
blk('html', '<div class="nld-page-band" style="background:#132433;padding:28px 32px;display:flex;align-items:center;'
    'justify-content:space-between;gap:24px;margin:0 0 32px">\n<div><p style="margin:0 0 10px;font-size:13px;line-height:16px;'
    'letter-spacing:0.12em;text-transform:uppercase;font-weight:600;color:#A9B6BF"><span style="display:inline-block;width:24px;'
    'height:4px;background:#D98A52;vertical-align:middle;margin-right:10px"></span>Nonlinear Dynamics Laboratory</p>\n'
    '<h1 style="color:#F7F6F2;font-size:36px;line-height:42px;margin:0">NCVD toolbox</h1></div>\n'
    '<img src="https://web.uri.edu/nld/wp-content/uploads/sites/1815/nld-hero-on-navy.jpg" alt="" width="96" height="96" '
    'style="width:96px;height:auto;flex:none">\n</div>')
fig('backbone_tracking.gif', 'Animation: sliding-window NCVD along a Duffing free decay traces frequency-energy dependence',
    'One free decay of a three-mass chain with a cubic spring. Each shaded window is decomposed; the three resonances trace '
    'their frequency-energy dependence back to the linear frequencies.')
p('Nonlinear Characteristic Value Decomposition: modal identification from a measured or simulated state record, with '
  'exact identities, pole correction for numerical derivatives, noise-robust operators, measured forcing, ensembles of '
  'records, and amplitude-dependent identification of nonlinear structures. MATLAB and Python, MIT license.')
blk('html', f'<p style="margin:8px 0 24px"><a href="{GH}" style="display:inline-block;background:#9A5323;color:#FFFFFF;'
    'padding:10px 16px;border-radius:2px;text-decoration:none;font-weight:600;margin:0 8px 8px 0">Code on GitHub</a>'
    f'<a href="{GH}/blob/main/docs/manual/NCVD_manual.pdf" style="display:inline-block;border:1px solid #8A8578;color:#132433;'
    'padding:9px 16px;border-radius:2px;text-decoration:none;margin:0 8px 8px 0">User manual (PDF)</a>'
    f'<a href="{GH}/raw/main/docs/slides/NCVD_toolbox.pptx" style="display:inline-block;border:1px solid #8A8578;color:#132433;'
    'padding:9px 16px;border-radius:2px;text-decoration:none;margin:0 8px 8px 0">Slides (PowerPoint)</a></p>')

# ---------------------------------------------------------------- decomposition
eyebrow('The decomposition')
h2('One state record, two identities, both exact')
p(f'With displacement, velocity and acceleration at {m("n")} stations, the toolbox writes the state record and its rate as')
blk('html', m(r'\mathbf{X}_1=[\mathbf{q},\ \dot{\mathbf{q}}]=\mathbf{U}\boldsymbol{\Phi}^{\mathsf{H}},\qquad '
              r'\mathbf{X}_2=[\dot{\mathbf{q}},\ \ddot{\mathbf{q}}]=\mathbf{U}\boldsymbol{\Lambda}\boldsymbol{\Phi}^{\mathsf{H}}+\mathbf{R}', True))
p(f'where {m(r"\mathbf{U}")} is the temporal basis, {m(r"\boldsymbol{\Phi}")} the characteristic shapes, '
  f'{m(r"\boldsymbol{\Lambda}=\operatorname{diag}(\lambda_i)")} the characteristic values (poles) and {m(r"\mathbf{R}")} '
  'whatever no linear model of the record explains: noise, unmeasured forcing, nonlinearity. The first identity is always '
  'exact; the second is exact for a clean linear record, and each pole is paired with its shape by construction.')
fig('complex_mode.gif', 'Animation: a complex mode as a rotating, shrinking polygon of phasors',
    'With non-proportional damping a mode is complex: the station phasors form a polygon that rotates and shrinks, and the '
    'motion is its real part, so the nodes travel.')

# ---------------------------------------------------------------- readout
eyebrow('The readout')
h2('Rotating by the eigenvectors of the coupling matrix makes the decomposition exact')
p('The decomposition builds on the characteristic value decomposition (CVD/GCVD) of Li, Stein and Chelidze (2025). '
  f'It forms the same balanced quotient of second moments and its eigenvectors {m(r"\mathbf{V}")}, then diagonalises the '
  'coupling matrix and rotates:')
blk('html', m(r'\boldsymbol{\Sigma}_2=\mathbf{V}^{\mathsf{H}}\mathbf{B}\,\mathbf{V}^{-\mathsf{H}}=\mathbf{T}\boldsymbol{\Lambda}\mathbf{T}^{-1},\qquad '
              r'\mathbf{U}=\boldsymbol{\Xi}\,\mathbf{T},\qquad \boldsymbol{\Phi}=\mathbf{V}\,\mathbf{T}^{-\mathsf{H}}', True))
p(f'The values are then the eigenvalues of the least-squares state-rate operator {m(r"\mathbf{B}=\mathbf{X}_1^{\dagger}\mathbf{X}_2")}. '
  f'Reading them from the diagonal of {m(r"\boldsymbol{\Sigma}_2")} instead is exact only when '
  f'{m(r"\mathbf{V}")} already diagonalises it. The balanced quotient applies the Joukowski map '
  f'{m(r"\lambda\mapsto\tfrac12(\lambda+\lambda^{-1})")}, which folds the pole plane at the unit circle; averaged ensemble '
  'quotients and nonlinear windows also break diagonality. There the diagonal readout errs at second order in the poles '
  'and first order in the shapes.')
fig('joukowski_fold.gif', 'Animation: the Joukowski fold of the balanced quotient as the time unit changes',
    'Rescaling time moves the poles across the unit circle; where a pole meets its conjugate under the balanced map, the '
    'off-diagonal part of the coupling matrix spikes.')
fig('ex07_readout_error.jpg', 'Published against canonical readout errors',
    'Left: an exact tie at a unit natural frequency. Right: windows of a nonlinear decay; pole differences scale with the '
    'square of the off-diagonal ratio, shape differences with its first power.')

# ---------------------------------------------------------------- practical
eyebrow('Practical data')
h2('Derivatives, sensor noise, forcing and many records')
rows = [('Displacement, velocity and acceleration', 'ncvd(d, v, a)'),
        ('Displacement only', 'state_from_displacement, then correct_poles'),
        ('White sensor noise, sampling well above the highest mode', "operator='bcls', noise_var='auto'"),
        ('Measured excitation', 'forcing=F'),
        ('Several records at any sampling rates', 'ncvd_ensemble(records)'),
        ('Spectrum spanning decades', 'rank=r'),
        ('Nonlinear free decay', 'track(d, v, a, ...)')]
blk('table', '<figure class="wp-block-table"><table><thead><tr><th>Your data</th><th>Use</th></tr></thead><tbody>'
    + ''.join(f'<tr><td>{a}</td><td><code>{html.escape(b)}</code></td></tr>' for a, b in rows)
    + '</tbody></table></figure>', '{"hasFixedLayout":false}')
p('Rates computed from displacement by any FIR stencil return the stencil symbol of the pole, not the pole itself:')
blk('html', m(r'\tilde\lambda=d\!\left(e^{\lambda\Delta t}\right),\qquad d(\theta)=f_s\sum_j c_j\,\theta^{j},\qquad \lambda=f_s\log\theta', True))
p('Inverting the symbol recovers the pole to machine precision for forward, central, five-point and Savitzky–Golay '
  'stencils, inside each stencil\'s band.')
fig('ex02_fd_correction.jpg', 'Pole errors from FIR derivatives before and after correction',
    'Three-mass chain with modes at 4.2, 71 and 158 rad/s. Solid: poles from FIR rates. Dotted: after correction.')
fig('ex03_noise_operators.jpg', 'Frequency and damping errors under sensor noise for three operators',
    'Ten-mass chain under unmeasured random forcing with noise on every channel. Bias-compensated least squares keeps the '
    'frequency error below 2 % down to 15 dB, where least squares reaches 10 %.')

# ---------------------------------------------------------------- nonlinear
eyebrow('Nonlinear systems')
h2('Sliding windows trace frequency-energy dependence from one free decay')
p('Each short window of a decay is decomposed; its values are the best linear description of the local dynamics. '
  'Following each resonance through the windows gives the data-driven counterpart of a nonlinear normal mode backbone curve, '
  'and the window residual shows how strongly each window engages the nonlinearity.')
fig('ex05_duffing_backbone.jpg', 'Frequency against modal energy for three tracked resonances',
    'Damped frequency against modal energy, normalised by the linear value; crosses are the measured response frequency of '
    'mode 1 in each window.')
items = ['<strong>Excite every resonance you track</strong>; check the singular values of each window.',
         '<strong>Span at least one period</strong> of the slowest mode per window; a dozen is comfortable.',
         '<strong>Watch weak modes</strong>: a superharmonic of a strong mode can take a weak mode\'s pole slot.',
         '<strong>Validate every pole</strong> against the measured response frequency (<code>band_frequency</code>).',
         '<strong>Trust frequencies and shapes</strong>; damping from single nonlinear windows scatters.']
blk('list', '<ol class="wp-block-list">' + ''.join(f'<!-- wp:list-item -->\n<li>{t}</li>\n<!-- /wp:list-item -->' for t in items)
    + '</ol>', '{"ordered":true}')

# ---------------------------------------------------------------- get started
eyebrow('Get started')
h2('Using the code')
p('Python (NumPy and SciPy):')
code("""git clone https://github.com/nld-uri/ncvd-toolbox.git
cd ncvd-toolbox/python && pip install -e .

import ncvd as nc
rec = nc.ncvd(d, v, a)                  # X1 = U Phi^H, X2 = U Lam Phi^H + R
mp  = nc.modal_parameters(rec.lam)      # wn, wd, zeta
tr  = nc.track(d, v, a, fs=fs, length=800, step=400, M=M)   # backbones""")
p('MATLAB (base MATLAB, R2022a or later):')
code("""addpath('ncvd-toolbox/matlab')
rec = ncvd.decompose(d, v, a);
mp  = ncvd.modalParameters(rec.lam);
tr  = ncvd.track(d, v, a, Fs=fs, Length=800, Step=400, M=M);""")
p(f'Both packages share the same API and are tested against each other. The <a href="{GH}/blob/main/docs/manual/NCVD_manual.pdf">user manual</a> '
  f'covers the theory, the options and seven worked examples; the <a href="{GH}">repository README</a> lists the contents. '
  'If you use the toolbox, please cite H.-W.-X. Li, D. L. Stein, D. Chelidze, <em>Mechanical Systems and Signal Processing</em> '
  '222 (2025) 111769, and D. L. Stein, D. Chelidze, <em>Nonlinear Dynamics</em> 113 (2025) 25–45.')

out = '\n\n'.join(B)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'page.html'), 'w') as f:
    f.write(out)
print(len(out), 'chars,', len(B), 'blocks')
