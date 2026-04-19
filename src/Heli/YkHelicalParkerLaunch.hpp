#pragma once
#include <cuda_runtime_api.h>

namespace YK {
	namespace Helical {

		void helical_parker_upload(
			const float* h_angles,
			int          K,
			float        angle_base);

		void helical_parker_launch(
			float* d_data,
			int Nu, int Nv, int K,
			float fSDD,
			float fDetUSize,
			float fCentralFanAngle,
			float fScale,
			cudaStream_t stream);

	} // namespace Helical
} // namespace YK