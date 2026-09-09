#pragma once

// Umbrella header — include this file to pull in the entire
// filter-kernel subsystem.
//
// Namespace layout:
//
//   YK::Filter          FilterKernelFFT (public API)
//   YK::Filter::detail  __global__ kernels + device helpers (internal)
//
// Typical usage:
//
//   #include "YkFilterKernel.cuh"
//
//   YK::Filter::FilterKernelFFT fk;
//   fk.prepare(paddedN, stream);
//   auto d_w = fk.alloc_weights();
//   fk.build_weights(d_w, desc);

#include "YkCreateFilterKernelHelpers.cuh"
#include "YkCreateFilterKernelFFT.hpp"
