"""Build the earlier template-based deck (NCVD_toolbox_v1.pptx) from the NLD template with LaTeX equations and MP4 animations.

Superseded: NCVD_toolbox.pptx is now exported from the live slides by live/lib/export_pptx.py.

    python build_deck.py

Equations are typeset by LaTeX (standalone class, Computer Modern) and placed
as 600-dpi PNG; animations come from ../animations; figures from
../manual/figures. Speaker notes carry the details.
"""
import os, re, subprocess, shutil, copy
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, 'template', 'NLD-example-deck.pptx')
ANIM = os.path.join(HERE, '..', 'animations')
FIGS = os.path.join(HERE, '..', 'manual', 'figures')
EQ = os.path.join(HERE, 'eq')
os.makedirs(EQ, exist_ok=True)
INK = '132433'

PRE = r"""\documentclass[border=2pt]{standalone}
\usepackage{amsmath,xcolor}\usepackage{fontspec}\usepackage{unicode-math}
\setmainfont{IBM Plex Sans}\setmathfont{latinmodern-math.otf}\setmathrm{Latin Modern Roman}
\definecolor{ink}{HTML}{132433}\definecolor{cu}{HTML}{B4622A}
\newcommand{\X}{\symbf{X}}\newcommand{\U}{\symbf{U}}\newcommand{\V}{\symbf{V}}
\newcommand{\T}{\symbf{T}}\newcommand{\B}{\symbf{B}}\newcommand{\R}{\symbf{R}}
\newcommand{\Y}{\symbf{Y}}\newcommand{\K}{\symbf{K}}\newcommand{\E}{\symbf{E}}\newcommand{\F}{\symbf{F}}
\newcommand{\Ph}{\symbf{\Phi}}\newcommand{\Lam}{\symbf{\Lambda}}\newcommand{\Sg}{\symbf{\Sigma}}
\newcommand{\Xib}{\symbf{\Xi}}\renewcommand{\H}{\mathsf{H}}
\begin{document}\color{ink}"""


def _render(name, src):
    tex = os.path.join(EQ, name + '.tex')
    with open(tex, 'w') as f:
        f.write(src)
    subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', name + '.tex'],
                   cwd=EQ, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['pdftocairo', '-png', '-r', '600', '-transp', '-singlefile',
                    name + '.pdf', name], cwd=EQ, check=True)
    return os.path.join(EQ, name + '.png')


EMU_PER_CM = 360000


def mtext(slide, name, lines, left, top, width, size=20, gap=8):
    """Sentences containing mathematics, typeset by XeLaTeX (Plex Sans text,
    Latin Modern math) at true size: the minipage is exactly the box width."""
    wcm = width/EMU_PER_CM
    body = (r'\par\vspace{%dpt}' % gap).join(lines)
    src = (PRE + r'\begin{minipage}{%.3fcm}\raggedright\fontsize{%d}{%d}\selectfont ' % (wcm, size, int(size*1.3))
           + body + r'\end{minipage}\end{document}')
    png = _render(name, src)
    from PIL import Image
    w, h = Image.open(png).size
    return slide.shapes.add_picture(png, left, top, width, int(width*h/w))


def eq(name, body, size=r'\Large'):
    """Render display math to EQ/name.png (transparent, 600 dpi); return path."""
    return _render(name, PRE + size + r'$\displaystyle ' + body + r'$\end{document}')


def pdf_png(pdf, out, dpi=220):
    subprocess.run(['pdftocairo', '-png', '-r', str(dpi), '-singlefile', pdf, out], check=True)
    return out + '.png'


def clear_slides(prs):
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst):
        prs.part.drop_rel(sldId.rId)
        sldIdLst.remove(sldId)


def layout(prs, name):
    return next(l for l in prs.slide_layouts if l.name == name)


def ph(slide, idx):
    return next(p for p in slide.placeholders if p.placeholder_format.idx == idx)


def set_text(shape, lines, size=None, bold_first=False):
    tf = shape.text_frame
    tf.clear()
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln
        if size:
            for r in p.runs:
                r.font.size = Pt(size)
        if bold_first and i == 0:
            for r in p.runs:
                r.font.bold = True


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def add_picture_fit(slide, path, left, top, max_w, max_h, align='center'):
    from PIL import Image
    w, h = Image.open(path).size
    s = min(max_w/w, max_h/h)
    W, H = int(w*s), int(h*s)
    x = left + (max_w - W)//2 if align == 'center' else left
    return slide.shapes.add_picture(path, x, top, W, H)


def add_eq(slide, path, left, top, height=None, width=None, max_w=None):
    from PIL import Image
    w, h = Image.open(path).size
    if height is not None:
        W = int(height*w/h); H = height
        if max_w and W > max_w:
            W = max_w; H = int(W*h/w)
    else:
        W = width; H = int(width*h/w)
    return slide.shapes.add_picture(path, left, top, W, H)


def add_movie(slide, name, left, top, max_w, max_h):
    mp4 = os.path.join(ANIM, name + '.mp4')
    poster = pdf_png(os.path.join(ANIM, name + '_still.pdf'), os.path.join(EQ, 'poster_' + name), 150)
    aspect = 1600/700
    W = min(max_w, int(max_h*aspect)); H = int(W/aspect)
    x = left + (max_w - W)//2
    return slide.shapes.add_movie(mp4, x, top, W, H, poster_frame_image=poster, mime_type='video/mp4')


def textbox(slide, left, top, w, h, lines, size=18, color=INK, font='IBM Plex Sans'):
    tb = slide.shapes.add_textbox(left, top, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = ln
        r.font.size = Pt(size); r.font.name = font; r.font.color.rgb = RGBColor.from_string(color)
        p.space_after = Pt(6)
    return tb


# ------------------------------------------------------------------ build
prs = Presentation(TPL)
clear_slides(prs)
SW, SH = prs.slide_width, prs.slide_height
L = 609905; CW = 10971886          # content left and width from the template
TOP = 2011680; BOT = 6217920        # content band

def content(eyebrow, title, layout_name='NLD Title Only'):
    s = prs.slides.add_slide(layout(prs, layout_name))
    ph(s, 101).text = eyebrow.upper()
    ph(s, 102).text = title
    return s

def section(num, title):
    s = prs.slides.add_slide(layout(prs, 'NLD Section'))
    ph(s, 101).text = num; ph(s, 102).text = title
    return s

# 1 title
s = prs.slides.add_slide(layout(prs, 'NLD Title'))
ph(s, 103).text = 'NCVD: an exact characteristic value decomposition'
ph(s, 104).text = 'Toolbox for Python and MATLAB \u00b7 github.com/nld-uri/ncvd-toolbox \u00b7 Nonlinear Dynamics Laboratory, URI'
notes(s, 'NCVD toolbox, version 1.0. Python package and MATLAB package with identical APIs, a manual, and these slides.')

# 2 two identities
s = content('The decomposition', 'One state record, two identities, both exact')
add_eq(s, eq('identities', r'\begin{aligned}\X_1&=[\symbf{q},\ \dot{\symbf{q}}]=\U\Ph^{\H}\\[4pt]'
                             r'\X_2&=[\dot{\symbf{q}},\ \ddot{\symbf{q}}]=\U\Lam\Ph^{\H}+\R,\qquad \Lam=\operatorname{diag}(\lambda_i)\end{aligned}'),
       L, TOP + 100000, height=1250000)
mtext(s, 'txt_identities', [
    r'$\U$: temporal basis. $\Ph$: characteristic shapes. $\lambda_i$: characteristic values (poles).',
    r'$\R$: what the diagonal form cannot represent: noise, unsupplied forcing, nonlinearity.',
    r'Linear, noise-free record: $\R=\symbf{0}$ to machine precision, and $\lambda_i$ belongs to column $i$ of $\Ph$ by construction.'],
    L, TOP + 1650000, CW)
notes(s, 'Rows are time instants. For a linear system X2 = X1 A^T. The first identity is always exact; the second is exact for clean linear data and is the least-squares remainder otherwise.')

# 3 complex mode
s = content('Geometry', 'A complex mode is a rotating, shrinking polygon')
add_movie(s, 'complex_mode', L, TOP - 100000, CW, BOT - TOP + 50000)
notes(s, 'Each station contributes a phasor psi_j exp(lambda t). With non-proportional damping the phases differ, so the polygon rotates at Im lambda and shrinks at Re lambda; the motion is its real part and the nodes travel. MPC measures how close the polygon is to a line (0.52 here). 6-DOF chain with dampers at masses 1 and 2.')

# 4 residual
s = content('Geometry', 'The residual is what no linear model of the record explains')
add_movie(s, 'residual_projection', L, TOP - 100000, CW, BOT - TOP + 50000)
notes(s, 'Each rate column splits into its projection on span X1, reproduced exactly by U Lam Phi^H, and a remainder orthogonal to it. For a window of a nonlinear decay ||R||/||X2|| indicates how strongly the window engages the nonlinearity.')

# 5 section
section('01', 'The readout')

# 6 published algorithm
s = content('The readout', 'The published GCVD reads poles from the diagonal of the coupling matrix')
add_eq(s, eq('algorithm', r'\begin{aligned}&\Q=\tfrac14\big(\K_{21}\K_{11}^{-1}+\K_{22}\K_{12}^{-1}+\K_{11}\K_{21}^{-1}+\K_{12}\K_{22}^{-1}\big),\quad \Q\V=\V\symbf{M}\\[3pt]'
                            r'&\Xib=\Y_1\V^{-\H},\qquad \Sg_2=\V^{\H}\B\,\V^{-\H},\qquad \B=\Y_1^{\dagger}\Y_2\\[3pt]'
                            r'&\text{published:}\quad r_i=(\Sg_2)_{ii},\quad \text{shapes } \V\end{aligned}'.replace(r'\Q', r'\symbf{Q}')),
       L, TOP + 150000, height=1500000, max_w=CW)
mtext(s, 'txt_published', [
    r'Exact only when $\V$ diagonalises $\Sg_2$, that is, when $\V$ are eigenvectors of $\B^{\H}$.',
    r'True for the CVD and \textquotedblleft forced\textquotedblright{} quotients on any data, and for the balanced quotient on exact linear data.',
    r'Not true for averaged ensemble quotients, nonlinear windows, or exact Joukowski ties.'],
    L, TOP + 1950000, CW)
notes(s, 'Li, Stein and Chelidze, MSSP 222 (2025) 111769, Algorithm 1 and Eqs. 13-14.')

# 7 Joukowski
s = content('The readout', 'The balanced quotient folds the pole plane at the unit circle')
add_movie(s, 'joukowski_fold', L, TOP - 100000, CW, BOT - TOP + 50000)
notes(s, 'For exact linear data Q = (A + A^-1)/2, so mu = (lambda + 1/lambda)/2: a Joukowski map, one-to-one only on one side of the unit circle. Rescaling time by c moves the poles; when a pole crosses |lambda/c| = 1 it meets its conjugate and V is undetermined. The off-diagonal of Sigma_2 spikes exactly at c = |lambda_k|. time_scale=auto places all poles at modulus >= 2.')

# 8 readout error
s = content('The readout', 'Off the diagonal, poles err at second order and shapes at first order')
add_picture_fit(s, pdf_png(os.path.join(FIGS, 'ex07_readout_error.pdf'), os.path.join(EQ, 'fig_ex07'), 250),
                L, TOP - 50000, CW, BOT - TOP)
notes(s, 'Left: exact tie, omega = 1, published pole error 2.5 at c = 1, machine precision at |c-1| = 1e-4. Right: windows of a Duffing decay; fitted slopes 1.74 (poles) and 1.05 (shapes). Shape-derived ROM quantities inherit the first-order change.')

# 9 canonical
s = content('The readout', 'Rotating by the eigenvectors of the coupling matrix makes both identities exact')
add_eq(s, eq('canonical', r'\Sg_2=\T\Lam\T^{-1},\qquad \U=\Xib\,\T,\qquad \Ph=\V\,\T^{-\H}'), L, TOP, height=520000)
add_eq(s, eq('equivalence', r'\E=\V^{-\H}\T:\qquad \B\E=\E\Lam,\qquad \U=\Y_1\E,\qquad \Ph=\E^{-\H},\qquad \R=(\symbf{I}-\Y_1\Y_1^{\dagger})\Y_2'),
       L, TOP + 900000, height=520000, max_w=CW)
textbox(s, L, TOP + 1800000, CW, 2200000, [
    'Characteristic values are the eigenvalues of the least-squares operator: the state-variable decomposition with a least-squares rate model.',
    'The balanced quotient fixes only the normalisation and the conditioning of the intermediate steps.',
    'No pole-to-shape assignment step is needed.'], size=20)
notes(s, 'Proposition 1 of the manual. With the default LS operator the canonical poles are least-squares poles; the contribution is the exactness analysis and the toolset around it.')

# 10 section
section('02', 'Practical data')

# 11 FIR
s = content('Practical data', 'Any FIR derivative stencil can be inverted inside its band')
add_movie(s, 'fir_band', L, TOP - 100000, int(CW*0.66), BOT - TOP + 50000)
add_eq(s, eq('fir', r'\begin{aligned}\tilde\lambda&=d\!\left(e^{\lambda\Delta t}\right)\\ d(\theta)&=f_s\textstyle\sum_j c_j\theta^{j}\\ \lambda&=f_s\log\theta\end{aligned}', size=r'\large'),
       L + int(CW*0.70), TOP + 300000, height=1300000)
mtext(s, 'txt_fir', [
    r'Machine precision for forward, central, five-point and Savitzky--Golay rates.',
    r'Invertible bands: $\omega\Delta t<\pi$, $\pi/2$, $1.80$, $1.14$.'],
    L + int(CW*0.70), TOP + 1900000, int(CW*0.30), size=16)
notes(s, 'With X1 = [q, Dq], X2 = [Dq, DDq] and one stencil D, a linear record satisfies X2 = X1 A~^T exactly with eigenvalues d(theta). correct_poles solves d(theta) = lambda~ and keeps the lowest-frequency root. Forward difference reproduces Eq. 48 of the paper. Central-type stencils turn over, so the inversion is unique only inside the band.')

# 12 noise
s = content('Practical data', 'Bias-compensated least squares removes the noise bias in frequency')
add_picture_fit(s, pdf_png(os.path.join(FIGS, 'ex03_noise_operators.pdf'), os.path.join(EQ, 'fig_ex03'), 250),
                L, TOP - 50000, int(CW*0.68), BOT - TOP)
add_eq(s, eq('bcls', r'\B=\left(\K_{11}-m\,\symbf{N}\right)^{-1}\K_{12}', size=r'\large'),
       L + int(CW*0.71), TOP + 300000, height=330000)
mtext(s, 'txt_bcls', [
    r'$\symbf{N}$: white sensor-noise variances, estimated from the flat spectrum near Nyquist.',
    r'At 15\,dB: frequency error 10\,\% (LS) against 1.9\,\% (BCLS).',
    r'TLS fails under unmeasured forcing.'],
    L + int(CW*0.71), TOP + 900000, int(CW*0.29), size=16)
notes(s, '10-DOF chain of the paper under white forcing at every mass, 200 Hz, 150 s, independent noise on every channel, 10 records per SNR. Remaining damping error (11-16 %) is the statistical error of a finite record under random forcing.')

# 13 forcing and ensembles
s = prs.slides.add_slide(layout(prs, 'NLD Two Column'))
ph(s, 101).text = 'PRACTICAL DATA'; ph(s, 102).text = 'Measured forcing and many records need no extra model'
set_text(ph(s, 103), ['Measured forcing', 'Project the input record out of both blocks.',
                      'No mass or input matrix needed.', 'Pole error at machine precision with the forcing, 0.048 rad/s without.'], bold_first=True)
set_text(ph(s, 104), ['Ensembles', 'Stack the rows of every record and decompose once.',
                      'Any sampling rates and lengths: the state-rate relation is continuous-time.',
                      '16 trials: frequency error from 1.5 % to 0.34 %.'], bold_first=True)
add_eq(s, eq('forcing', r'\X_2=\X_1\B+\F\symbf{G}+\R', size=r'\normalsize'), L, BOT - 450000, height=330000)
add_eq(s, eq('ensemble', r'\begin{bmatrix}\X_2^{(1)}\\ \vdots\\ \X_2^{(N)}\end{bmatrix}\approx\begin{bmatrix}\X_1^{(1)}\\ \vdots\\ \X_1^{(N)}\end{bmatrix}\B', size=r'\normalsize'),
       6324905, BOT - 1050000, height=950000)
notes(s, 'Forcing: Frisch-Waugh projection, then the operator; gains returned as G. Ensemble: least-squares fusion replaces the quotient averaging of Eq. 59 and is exact for clean data. Weights: equal per record (default), per sample, or by SNR.')

# 14 section
section('03', 'Nonlinear systems')

# 15 backbone
s = content('Nonlinear systems', 'Sliding windows trace frequency-energy dependence from one decay')
add_movie(s, 'backbone_tracking', L, TOP - 100000, CW, BOT - TOP + 50000)
notes(s, 'Duffing chain, cubic spring grounding mass 1, damping 0.003 M, 1500 s decay, 40 s windows every 10 s. All three backbones converge to the linear frequencies (0.770/0.765, 1.419/1.414, 1.850/1.848). Mode-1 poles agree with the measured response frequency to 1.5 %. The window residual falls with energy.')

# 16 caveats
s = content('Nonlinear systems', 'Each window is an equivalent-linear fit, and it needs the right data', 'NLD Content')
set_text(ph(s, 103), [
    'Excite every resonance you track; check the singular values of each window.',
    'Span at least one period of the slowest mode; a dozen is comfortable.',
    'A weak mode can lose its pole slot to a superharmonic of a strong one.',
    'Validate every pole against the measured response frequency.',
    'Trust frequencies and shapes; single-window damping scatters.',
    'Shape-derived ROM quantities change at first order with the readout: do not mix readouts.'])
notes(s, 'See Section 4.2 of the manual.')

# 17 usage
s = prs.slides.add_slide(layout(prs, 'NLD Two Column'))
ph(s, 101).text = 'USING THE TOOLBOX \u00b7 GITHUB.COM/NLD-URI/NCVD-TOOLBOX'; ph(s, 102).text = 'The same calls in Python and MATLAB'
for idx, lines in [(103, ['Python', 'import ncvd as nc', 'rec = nc.ncvd(d, v, a)', 'mp = nc.modal_parameters(rec.lam)',
                          "tr = nc.track(d, v, a, fs=fs, length=800, step=400, M=M)", 'pytest: 29 tests']),
                   (104, ['MATLAB', 'rec = ncvd.decompose(d, v, a);', 'mp = ncvd.modalParameters(rec.lam);',
                          'tr = ncvd.track(d, v, a, Fs=fs, Length=800, Step=400, M=M);', "runtests: 14 tests, Python cross-checks"])]:
    tf = ph(s, idx).text_frame; tf.clear()
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = ln
        if i == 0:
            r.font.bold = True
        elif i < len(lines) - 1:
            r.font.name = 'IBM Plex Mono'; r.font.size = Pt(15)
notes(s, 'Options: rank, center, operator (ls, tls, bcls), forcing, time_scale, weights, readout (eig or diag for benchmarking). Manual: docs/manual/NCVD_manual.pdf.')

# 18 closing
s = prs.slides.add_slide(layout(prs, 'NLD Closing'))
ph(s, 104).text = 'github.com/nld-uri/ncvd-toolbox \u00b7 web.uri.edu/nld/ncvd-toolbox\nchelidze@uri.edu'

out = os.path.join(HERE, 'NCVD_toolbox_v1.pptx')
prs.save(out)
print('wrote', out, len(prs.slides._sldIdLst), 'slides')
