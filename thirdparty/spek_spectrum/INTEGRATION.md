# Vendored spek_spectrum

Imported on 2026-10-11 from mcgpu-modern/third_party/spek_spectrum (version 0.1.0).
Numerical models and embedded tables derive from SpekPy 2.5.4; retain LICENSE.
The vendored numerical implementation and table bytes are unchanged.
The generated table array is split into source .inc chunks below 2 MB each
to comply with the repository file-size hook; compilation still embeds one array. Only include/, src/ and
CMakeLists.txt are needed for normal builds. Python scripts are maintainer tools.
The library is linked only by the multispectrum tool/tests, not the YKCBCT DLL.

The adapter feeds center energies and explicit widths directly to SpectrumPoint,
normalized to 1 mAs at 100 cm. YK-SPECTRUM exports preserve those centers/widths.
Tagged MCGPU-SPEC imports convert lower edges plus the final zero upper edge to
centers; untagged existing SpekCalc-style input retains its existing interpretation.
