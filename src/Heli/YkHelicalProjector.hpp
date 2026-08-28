#pragma once
#include <vector>
#include "common/YkProjectionOperators.hpp"
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
			SReconstructionParams fp_params{};
			fp_params.scan.Nu = param_.iPU;
			fp_params.scan.Nv = param_.iPV;
			fp_params.scan.NAng = total_views;
			fp_params.scan.totalViews = total_views;
			fp_params.scan.sid_mm = param_.SID;
			fp_params.scan.sdd_mm = param_.SDD;
			fp_params.scan.du_mm = param_.du_mm;
			fp_params.scan.dv_mm = param_.dv_mm;
			fp_params.scan.angles = param_.angle_list;
			fp_params.volume.Nx = param_.iVX;
			fp_params.volume.Ny = param_.iVY;
			fp_params.volume.Nz = param_.iVZ;
			fp_params.volume.voxelX_mm = param_.vox_x_mm;
			fp_params.volume.voxelY_mm = param_.vox_y_mm;
			fp_params.volume.voxelZ_mm = param_.vox_z_mm;
			fp_params.volume.centerX_mm = param_.vol_offset_x_mm;
			fp_params.volume.centerY_mm = param_.vol_offset_y_mm;
			fp_params.volume.centerZ_mm = param_.vol_offset_z_mm;

			GeometryContext geometry;
			ResourceContext resources;
			if (!geometry.initialize(fp_params, geo_)) {
				cudaFree(d_proj);
				return false;
			}
			resources.attach(stream, device_id_);
			auto fp = makeForwardOperator(param_.fp_task);
			if (!fp->prepare(geometry, resources)) {
				cudaFree(d_proj);
				return false;
			}

			{
				Util::CudaTimer timer("helical_fp", stream);
				if (!fp->apply(d_vol, fp_params, d_proj, resources)) {
					fp->release();
					cudaFree(d_proj);
					return false;
				}
			}
			fp->release();

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
