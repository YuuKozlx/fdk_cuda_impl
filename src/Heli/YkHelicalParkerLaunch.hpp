#pragma once
#include <cuda_runtime_api.h>

namespace YK {
	namespace Helical {

		void helical_parker_upload(
			const float* h_angles,
			const float* h_z_src,
			int          K,
			float        angle_base);

		void helical_parker_launch(
			float* d_data,
			int Nu, int Nv, int K,
			float fSDD,
			float fSID,
			float fDetUSize,
			float fDetVSize,
			float fCentralFanAngle,
			float fScale,
			float z0,
			float z_half_range,
			cudaStream_t stream);

	} // namespace Helical
} // namespace YK