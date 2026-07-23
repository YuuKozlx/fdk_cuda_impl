
#include <algorithm>
#include <vector>
#include "global/YkGlobals.h"
#include "global/YkMacro.hpp"
#include "global/YkWarpStrideCtx.cuh"
#include "YkHelicalParkerLaunch.hpp"

namespace YK {
	namespace Helical {
		namespace detail {

			__constant__ float gC_helical_parker_angle[kMaxChunkAng];

			__global__ void helical_parker_kernel(
				float* __restrict__ data,
				int   Nu, int Nv, int K,
				float fSDD,
				float fDetUSize,
				float fCentralFanAngle,
				float fScale)
			{
				WarpStrideCtx<EWarpStrideAxis::RowWarp> ctx;
				const int total_rows = K * Nv;

				for (int row = ctx.warp_global; row < total_rows;
					row += ctx.n_warps)
				{
					const int angle = row / Nv;
					const int v = row - angle * Nv;
					if (angle >= K) continue;

					const float beta = gC_helical_parker_angle[angle];

					for (int u = ctx.lane; u < Nu; u += 32)
					{
						const float u_mm = (u - 0.5f * Nu + 0.5f) * fDetUSize;
						const float gamma = atanf(u_mm / fSDD);

						const float t1 = 2.0f * (fCentralFanAngle + gamma);
						const float t2 = CUDA_PI + 2.0f * gamma;
						const float t3 = CUDA_PI + 2.0f * fCentralFanAngle;

						float w_parker;
						if (beta <= 0.f) { w_parker = 0.f; }
						else if (beta < t1) {
							const float s = sinf(
								(CUDA_PI * 0.25f) * beta
								/ (fCentralFanAngle + gamma));
							w_parker = s * s;
						}
						else if (beta <= t2) { w_parker = 1.f; }
						else if (beta < t3) {
							const float s = sinf(
								(CUDA_PI * 0.25f)
								* (CUDA_PI + 2.f * fCentralFanAngle - beta)
								/ (fCentralFanAngle - gamma));
							w_parker = s * s;
						}
						else { w_parker = 0.f; }

						data[(angle * Nv + v) * Nu + u] *= w_parker * fScale;
					}
				}
			}

		} // namespace detail

		// ----------------------------------------------------------------
		// upload
		// ----------------------------------------------------------------
		void helical_parker_upload(
			const float* h_angles,
			int          K,
			float        angle_base)
		{
			std::vector<float> rel(K);
			for (int i = 0; i < K; ++i) {
				float f = h_angles[i] - angle_base;
				while (f < 0.f)             f += 2.f * CUDA_PI;
				while (f >= 2.f * CUDA_PI)  f -= 2.f * CUDA_PI;
				rel[i] = f;
			}
			YK_CUDA_CHECK(cudaMemcpyToSymbol(
				detail::gC_helical_parker_angle,
				rel.data(),
				K * sizeof(float), 0, cudaMemcpyHostToDevice));
		}


		// ----------------------------------------------------------------
		// launch
		// ----------------------------------------------------------------
		void helical_parker_launch(
			float* d_data,
			int Nu, int Nv, int K,
			float fSDD,
			float fDetUSize,
			float fCentralFanAngle,
			float fScale,
			cudaStream_t stream)
		{
			SKernelLaunchPolicy policy;
			policy.block_threads = 128;
			const auto launch = policy.makeRowWarp((size_t)K * Nv);

			detail::helical_parker_kernel << <launch.grid, launch.block, 0, stream >> > (
				d_data, Nu, Nv, K,
				fSDD, fDetUSize,
				fCentralFanAngle, fScale);

			YK_CUDA_KERNEL_CHECK();
		}

	} // namespace Helical
} // namespace YK
