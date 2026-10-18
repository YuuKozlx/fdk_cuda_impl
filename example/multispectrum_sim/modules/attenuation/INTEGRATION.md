# Project-maintained attenuation component

This component was previously stored in `thirdparty/nist_xcom`. It is maintained
as part of the multispectrum application: the C API, embedded data integration,
runtime configuration and error handling are part of the integration surface.
The numerical implementation and all embedded/reference data were moved without
regeneration. Original licensing and third-party notices remain in this directory.

`nist_xcom_core` builds `xcom.cpp` and `xcom_c_api.cpp` as one static library.
The application adapter and material definitions remain in
`src/input/materials/XcomAttenuationProvider.cpp` and the matching public headers.
Neither the reconstruction DLL nor its lower-level kernels depend on this module.
