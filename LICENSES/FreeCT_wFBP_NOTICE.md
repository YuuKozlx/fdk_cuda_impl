# FreeCT_wFBP attribution

The wFBP implementation under `src/Heli/wfbp` is adapted from the algorithm
and CUDA implementation in [FreeCT_wFBP](https://github.com/FreeCT/FreeCT_wFBP).

- Copyright (C) 2015 John Hoffman
- Original license: GNU General Public License, version 2 or (at your option)
  any later version (`GPL-2.0-or-later`)
- Adaptation copyright (C) 2026 YKCBCT contributors

The adaptation changes data structures, memory management, CUDA launch policy,
filter implementation and public interfaces, while retaining the FreeCT fan to
parallel rebinning and helical weighted-backprojection formulation.

The complete GPL-2.0 license text is available at:
https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt
