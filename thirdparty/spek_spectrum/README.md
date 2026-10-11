# spek_spectrum

Standalone C++17 implementation of the tungsten reflection-target
`spekcalc` and `spekpy-v1` models from SpekPy 2.5.4. It has no Python, CUDA,
MC-GPU, TOML, or SciPy runtime dependency.

The numerical model and packed float32 tables are derived from SpekPy, distributed
under the MIT license in `LICENSE`. The implementation covers spectrum
generation, tungsten self-filtration, elemental Be/Al/Cu/Sn/W filtration,
air kerma, mean energy, Al HVL, and MCGPU-SPEC output.

Named filter shortcuts cover Be, Ti, Al, Cu, Mo, Rh, Ag, Sn, W, Au, and Pb.
Any element in the embedded Z=1..92 attenuation data can be supplied using
`Filter::atomic_number` and `Filter::density_g_cm3`.

The packed data also contains the FITPACK knot and coefficient arrays created
by SpekPy's `RectBivariateSpline`. The read-only array is checked in as
`src/spek_tables_embedded.cpp` and linked directly into the library, so neither
building nor runtime evaluation requires Python, SciPy, or a data file.
`Config::data_file` remains available as an explicit external-table override.

`cmake/embed_spek_tables.cmake` is a maintainer-only regeneration tool. It is
not invoked by the normal CMake build.

`pack_spekpy_tables.py` writes float32 by default; `--precision float64` is
available for numerical comparison. The shipped float32 package is about
4.2 MB and agrees with the float64 package to roughly 1e-8 relative for the
120 kVp regression case.

```cpp
spek_spectrum::Config config;
config.model = spek_spectrum::Model::spekcalc;
config.kvp = 120.0;
config.anode_angle_deg = 12.0;
config.filters = {{"Al", 2.0}};
auto spectrum = spek_spectrum::generate(config);
```
