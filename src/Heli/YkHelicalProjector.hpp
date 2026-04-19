#pragma once
#include <vector>
#include "FP/YkFpRunnerExVec.hpp"
#include "Heli/YkHelicalGeo.hpp"
#include "Heli/YkHeliCTParams.h"
#include "global/YkLog.h"
#include "global/YkMacro.hpp"
#include "util/YkCudaTimer.hpp"

namespace YK {

	class HelicalProjector {
	public:

		bool init(const SHeliCTParam& param,
			cudaStream_t        stream,
			int                 device_id = 0)
		{
			if (param.angle_list.empty()) {
				YK_LOGE("[HelicalProjector] angle_list is empty");
				return false;
			}

			param_ = param;
			stream_ = stream;
			device_id_ = device_id;

			// 构建螺旋几何
			build_helical_vec_geometry(geo_, param_);

			is_initialized_ = true;
			YK_LOGI("[HelicalProjector] init OK: {} views  "
				"pitch={:.1f}mm  start_z={:.1f}mm",
				(int)geo_.size(),
				param_.pitch_mm, param_.start_z_mm);
			return true;
		}

		void release()
		{
			geo_.clear();
			is_initialized_ = false;
		}

		bool project(const float* d_vol,
			float* h_proj_out,
			cudaStream_t stream)
		{
			if (!is_initialized_) {
				YK_LOGE("[HelicalProjector] not initialized");
				return false;
			}

			const int    total_views = (int)geo_.size();
			const size_t view_elems =
				(size_t)param_.iPU * param_.iPV;
			const size_t proj_elems = view_elems * total_views;

			float* d_proj = nullptr;
			YK_CUDA_CHECK(cudaMalloc(
				&d_proj, proj_elems * sizeof(float)));
			YK_CUDA_CHECK(cudaMemset(
				d_proj, 0, proj_elems * sizeof(float)));

			// 构造 fp_params，从 SHeliCTParam 转换
			SCBCTParams fp_params{};
			fp_params.iPU = param_.iPU;
			fp_params.iPV = param_.iPV;
			fp_params.iVX = param_.iVX;
			fp_params.iVY = param_.iVY;
			fp_params.iVZ = param_.iVZ;
			fp_params.SID = param_.SID;
			fp_params.SDD = param_.SDD;
			fp_params.du_mm = param_.du_mm;
			fp_params.dv_mm = param_.dv_mm;
			fp_params.vox_x_mm = param_.vox_x_mm;
			fp_params.vox_y_mm = param_.vox_y_mm;
			fp_params.vox_z_mm = param_.vox_z_mm;
			fp_params.vol_offset_x_mm = param_.vol_offset_x_mm;
			fp_params.vol_offset_y_mm = param_.vol_offset_y_mm;
			fp_params.vol_offset_z_mm = param_.vol_offset_z_mm;
			fp_params.iPAng = total_views;
			fp_params.iPAngTotal = total_views;
			fp_params.angle_list = param_.angle_list;

			FpReconstructorEx fp;
			if (!fp.init(fp_params, param_.fp_task, device_id_)) {
				cudaFree(d_proj);
				return false;
			}

			{
				Util::CudaTimer timer("helical_fp", stream);
				if (!fp.run(d_vol, fp_params, geo_, d_proj, stream)) {
					cudaFree(d_proj);
					return false;
				}
			}

			YK_CUDA_CHECK(cudaStreamSynchronize(stream));
			YK_CUDA_CHECK(cudaMemcpy(
				h_proj_out, d_proj,
				proj_elems * sizeof(float),
				cudaMemcpyDeviceToHost));

			cudaFree(d_proj);
			return true;
		}

		const std::vector<SConeProjGeomVec>& geo() const { return geo_; }
		int  totalViews()    const { return (int)geo_.size(); }
		bool isInitialized() const { return is_initialized_; }

	private:
		bool          is_initialized_ = false;
		SHeliCTParam  param_{};
		cudaStream_t  stream_ = nullptr;
		int           device_id_ = 0;

		std::vector<SConeProjGeomVec> geo_;
	};

} // namespace YK