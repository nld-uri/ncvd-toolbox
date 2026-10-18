# Live slides and browser-native animations

The current deck is an HTML slide deck with live animations:
[NCVD toolbox slides](https://claude.ai/artifact/5XebWDvuxhZizJR5Ls6mmn).
A vector export of the same deck is `../NCVD_toolbox_slides.pdf`; animations appear there as one frame each.

`embeds/*.html` are the five animations as standalone pages. Open one in a browser to watch it.
Append `#p=0.4` to the URL to freeze it at 40 % of its cycle.

| File | Shows |
|---|---|
| `complex_mode.html` | a complex mode as a rotating, shrinking polygon of phasors |
| `residual_projection.html` | the rate column, its projection on span X1, and the residual R |
| `joukowski_fold.html` | the balanced quotient folding the pole plane at the unit circle |
| `fir_band.html` | FIR derivative symbols and the band inside which the pole correction is unique |
| `backbone_tracking.html` | sliding-window NCVD tracing backbone curves along a Duffing decay |

Rebuild: `python make_embeds.py` (needs the `ncvd` package in `../../../python`, numpy, scipy, and
`npx terser` for minification). Every page stays under 16 KB. `lib/plot.js` is the canvas plotting
kit; `lib/latexblock.py` typesets slide text blocks with XeLaTeX (IBM Plex Sans, Latin Modern Math);
`latex_spec.py` holds the typeset blocks used in the deck; `lib/export_pdf.py` renders a deck to PDF.
