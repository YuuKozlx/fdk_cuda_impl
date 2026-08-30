# FreeCT_wFBP attribution

The wFBP implementation under `src/Heli/analytic/wfbp` is adapted from the algorithm
and CUDA implementation in [FreeCT_wFBP](https://github.com/FreeCT/FreeCT_wFBP).

Reference revisions used for this port:

- FreeCT_wFBP legacy n/p/z/a FFS implementation:
  `83e6281eae4b1568fcb115133fac212a0f17f4a1`
- Unified FreeCT wFBP/filter implementation:
  `aa4bb3dd4c4f16de0ecfa0b3116f0829e7f20999`

- Copyright (C) 2015 John Hoffman
- Original license: GNU General Public License, version 2 or (at your option)
  any later version (`GPL-2.0-or-later`)
- Adaptation copyright (C) 2026 YKCBCT contributors

The adaptation changes data structures, memory management, CUDA launch policy,
FFT plumbing and public interfaces. The retained and ported algorithmic scope
includes the FreeCT n/p/z/a flying-focal-spot rebinning paths, the FreeCT
`r(t)` reconstruction filter, and the `W(q)` normalized helical
weighted-backprojection formulation. Flat-panel input and non-zero detector
V offset are integration extensions around the FreeCT core.

The complete GPL-2.0 license text is included in
`LICENSES/GPL-2.0-or-later.txt`.
