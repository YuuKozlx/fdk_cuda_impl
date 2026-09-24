# DLL Pipeline Example

`test_shepp_logan.cpp` demonstrates the public DLL interface without including
`src` headers or calling internal pipelines, CUDA allocation, or kernels.
The only non-SDK helper is the host-only BMP writer in `Yktest/YkTestImage.hpp`.

The test generates a modified 3-D Shepp-Logan phantom, executes
`EPipeline::ForwardProjection`, then executes ordinary flat `EPipeline::FDK`.
All buffers are caller-owned host `std::vector<float>` arrays. Sessions use
`ReconstructionSessionFactory::create/destroy` with RAII and report DLL errors.
After `reset`, the same FDK session reconstructs in 47-view batches, including
a short final batch, and compares the result with full-input reconstruction.

Build and run from a configured Visual Studio development environment:

```powershell
cmake --build out/build/x64-Release --target dlltest_shepp_logan
out/build/x64-Release/dlltest/dlltest_shepp_logan.exe
```

An optional first argument selects the artifact directory. The default is
`out/test-artifacts/dll-shepp-logan` relative to the working directory.
`comparison.bmp` shows the central axial slice: truth, FDK, absolute error,
all in the same `[0, 0.02]` window. `metrics.txt` records geometry, raw-value
NRMSE/MAE, cosine similarity, timings and the streaming maximum difference.
Timing includes initialization/transfers and is not a kernel benchmark.

This is a deterministic interface/numerical smoke test, not a claim of clinical
image quality. It fails for non-finite output, cosine similarity <= 0.85,
NRMSE >= 0.65, or full-vs-streaming maximum difference >= 1e-5.
