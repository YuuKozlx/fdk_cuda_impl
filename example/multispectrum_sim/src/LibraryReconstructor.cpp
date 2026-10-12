#include "common/YkTimingWatchdog.hpp"
#include "LibraryReconstructor.hpp"

#include "LibraryGeometry.hpp"
#include "ReconstructionPreprocessor.hpp"
#include "cuda/DeviceMemoryPool.hpp"
#include "recon/preprocess/ProjectionCache.hpp"
#include "YKCBCT/interface/YkReconstructionApi.hpp"
#include "Reconstruction/Iterative/Common/YkReconstructionMemoryBudget.hpp"
#include "Reconstruction/Iterative/Common/YkProjectionInput.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cuda_runtime_api.h>

namespace yk::spectral {
namespace {

void check(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void timedRead(std::ifstream& input,char* data,std::streamsize bytes){
    YK::diagnostics::TimingSpan watch("reconstruction.read","bytes="+std::to_string(bytes));input.read(data,bytes);
}
void timedWrite(std::ofstream& output,const char* data,std::streamsize bytes){
    YK::diagnostics::TimingSpan watch("reconstruction.write","bytes="+std::to_string(bytes));output.write(data,bytes);
}

void writeLe16(std::ofstream& out, std::uint16_t value)
{
    const char bytes[2] = {static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8u) & 0xffu)};
    out.write(bytes, 2);
}

void writeLe32(std::ofstream& out, std::uint32_t value)
{
    const char bytes[4] = {static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8u) & 0xffu),
        static_cast<char>((value >> 16u) & 0xffu),
        static_cast<char>((value >> 24u) & 0xffu)};
    out.write(bytes, 4);
}

// 写 Windows 常见的 8-bit 灰度 BMP。图像只是诊断输出，原始 float32
// 体数据仍是数值比较和后处理的唯一依据。
void writeBmp(const std::filesystem::path& path, int width, int height,
    const std::vector<float>& values, float low, float high)
{
    check(width > 0 && height > 0 && values.size() ==
        static_cast<std::size_t>(width) * height, "切片尺寸无效");
    if (!(high > low)) high = low + 1.f;
    const std::size_t row_bytes = (static_cast<std::size_t>(width) + 3u) & ~3u;
    const std::uint32_t pixel_bytes = static_cast<std::uint32_t>(row_bytes * height);
    std::ofstream out(path, std::ios::binary);
    check(static_cast<bool>(out), "无法创建切片图像: " + path.string());
    writeLe16(out, 0x4d42u);
    writeLe32(out, 14u + 40u + 256u * 4u + pixel_bytes);
    writeLe16(out, 0); writeLe16(out, 0);
    writeLe32(out, 14u + 40u + 256u * 4u);
    writeLe32(out, 40u); writeLe32(out, static_cast<std::uint32_t>(width));
    writeLe32(out, static_cast<std::uint32_t>(height));
    writeLe16(out, 1); writeLe16(out, 8); writeLe32(out, 0);
    writeLe32(out, pixel_bytes); writeLe32(out, 2835); writeLe32(out, 2835);
    writeLe32(out, 256); writeLe32(out, 0);
    for (std::uint32_t i = 0; i < 256; ++i) {
        const char color[4] = {static_cast<char>(i), static_cast<char>(i),
            static_cast<char>(i), 0};
        out.write(color, 4);
    }
    std::vector<char> row(row_bytes, 0);
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            const float t = std::clamp((values[static_cast<std::size_t>(y) * width + x] - low) /
                (high - low), 0.f, 1.f);
            row[static_cast<std::size_t>(x)] = static_cast<char>(std::lround(t * 255.f));
        }
        out.write(row.data(), static_cast<std::streamsize>(row.size()));
    }
    check(static_cast<bool>(out), "写入切片图像失败: " + path.string());
}

YK::EFdkFilter parseFilter(const std::string& name)
{
    if (name == "ramlak" || name == "ram-lak") return YK::EFdkFilter::RamLak;
    if (name == "cosine") return YK::EFdkFilter::Cosine;
    if (name == "hann") return YK::EFdkFilter::Hann;
    if (name == "hamming") return YK::EFdkFilter::Hamming;
    if (name == "shepp-logan") return YK::EFdkFilter::SheppLogan;
    throw std::runtime_error("未知滤波核: " + name);
}

YK::EPipeline parsePipeline(const std::string& name)
{
    if (name == "fdk") return YK::EPipeline::FDK;
    if (name == "wfbp") return YK::EPipeline::WFBP;
    if (name == "tigre_sart" || name == "tigre_sirt" ||
        name == "tigre_os_sart" || name == "tigre_sart_tv" ||
        name == "tigre_os_sart_tv")
        return YK::EPipeline::TigreGradient;
    if (name == "statistical") return YK::EPipeline::Statistical;
    if (name == "sart") return YK::EPipeline::SART;
    if (name == "sirt") return YK::EPipeline::SIRT;
    if (name == "ossart") return YK::EPipeline::OSSART;
    if (name == "cgls") return YK::EPipeline::CGLS;
    throw std::runtime_error("未知重建管线: " + name);
}

void writeSlices(const std::filesystem::path& prefix, const std::vector<float>& volume,
    int nx, int ny, int nz)
{
    float low = std::numeric_limits<float>::infinity();
    float high = -std::numeric_limits<float>::infinity();
    for (float v : volume) if (std::isfinite(v)) { low = std::min(low, v); high = std::max(high, v); }
    check(std::isfinite(low) && std::isfinite(high), "重建体没有有限数值");
    const int z = nz / 2, y = ny / 2, x = nx / 2;
    std::vector<float> axial(static_cast<std::size_t>(nx) * ny);
    std::vector<float> coronal(static_cast<std::size_t>(nx) * nz);
    std::vector<float> sagittal(static_cast<std::size_t>(ny) * nz);
    for (int yy = 0; yy < ny; ++yy) for (int xx = 0; xx < nx; ++xx)
        axial[static_cast<std::size_t>(yy) * nx + xx] = volume[(static_cast<std::size_t>(z) * ny + yy) * nx + xx];
    for (int zz = 0; zz < nz; ++zz) for (int xx = 0; xx < nx; ++xx)
        coronal[static_cast<std::size_t>(zz) * nx + xx] = volume[(static_cast<std::size_t>(zz) * ny + y) * nx + xx];
    for (int zz = 0; zz < nz; ++zz) for (int yy = 0; yy < ny; ++yy)
        sagittal[static_cast<std::size_t>(zz) * ny + yy] = volume[(static_cast<std::size_t>(zz) * ny + yy) * nx + x];
    writeBmp(prefix.string() + "-axial.bmp", nx, ny, axial, low, high);
    writeBmp(prefix.string() + "-coronal.bmp", nx, nz, coronal, low, high);
    writeBmp(prefix.string() + "-sagittal.bmp", ny, nz, sagittal, low, high);
}

} // namespace

bool reconstructWithLibraryFdk(const SimulationConfig& config,
    const std::filesystem::path& projection_file,
    const std::filesystem::path& volume_file,
    const std::filesystem::path& slice_prefix,
    std::string& diagnostic,TaskProgressCallback progress)
{
    YK::diagnostics::TimingSpan total_watch("reconstruction.run",config.reconstruction.iterative.algorithm);
    YK::IReconstructionSession* session = nullptr;
    std::filesystem::path reconstruction_input = projection_file;
    try {
        if (config.reconstruction.preprocessing.enabled) {
            std::string preprocessing_diagnostic;
            reconstruction_input=preprocess::cachedProjection(config,projection_file,
                [&](const std::filesystem::path& output,std::string& details){
                    return prepareReconstructionProjection(config,projection_file,output,details,progress);
                },preprocessing_diagnostic,progress);
            std::cout << preprocessing_diagnostic << std::endl;
        }
        const auto pool_before=yk::spectral::cuda_resources::devicePoolStatistics();
        {YK::diagnostics::TimingSpan watch("reconstruction.trim_pool");yk::spectral::cuda_resources::trimDevicePool();}
        std::cout<<"reconstruction_pool released_cached_bytes="<<pool_before.cached_bytes<<std::endl;
        const auto& g = config.geometry.parameters;
        const std::size_t pixels = static_cast<std::size_t>(g.detector_u) * g.detector_v;
    const std::size_t volume_elements = static_cast<std::size_t>(g.reconstruction_volume_x) *
        g.reconstruction_volume_y * g.reconstruction_volume_z;
        const std::size_t projection_elements = static_cast<std::size_t>(g.views) * pixels;
        std::ifstream input(reconstruction_input, std::ios::binary);
        check(static_cast<bool>(input), "无法读取投影文件: " + projection_file.string());
        input.seekg(0, std::ios::end);
        check(static_cast<std::uint64_t>(input.tellg()) == projection_elements * sizeof(float),
            "投影文件大小与 geometry_config 不匹配");
        input.seekg(0, std::ios::beg);

        const bool analytic = config.reconstruction.type == "analytic";
        const bool statistical=config.reconstruction.type=="statistical";
        const std::string pipeline_name = statistical?"statistical":analytic ? config.reconstruction.analytic.pipeline : config.reconstruction.iterative.algorithm;
        const bool fdk = pipeline_name == "fdk";
        const bool wfbp = pipeline_name == "wfbp";
        const bool cyl_fdk = fdk && config.geometry.kind == GeometryKind::CylCbct;
        const bool nonstandard_fdk = fdk &&
            (config.geometry.kind == GeometryKind::PlanarSourceOrbit ||
             config.geometry.kind == GeometryKind::NikonTilted);
        const bool iterative = !analytic;
        bool batched_projection=false;
        size_t planned_block=0,planned_batch_bytes=0;
        std::string projection_input_mode="auto";
        const auto pipeline = parsePipeline(pipeline_name);
        YK::SStatisticalWorkspaceEstimate statistical_workspace;
        check(fdk || wfbp || iterative, "不支持的 DLL 重建管线");
        check(!fdk || config.geometry.kind == GeometryKind::FlatCbct ||
            cyl_fdk || nonstandard_fdk,
            "DLL FDK requires a supported flat, cylindrical, or non-standard acquisition");
        check(!cyl_fdk || config.geometry.kind == GeometryKind::CylCbct,
            "DLL cyl_fdk 只放行 cyl_cbct");
        check(!wfbp || config.geometry.kind == GeometryKind::CylHelical,
            "DLL wFBP 当前只放行 cyl_helical");
        auto system = makeLibrarySystem(config, pipeline,
            YK::EProjectionModel::Joseph, parseFilter(config.reconstruction.analytic.filter),
            GeometryUse::Reconstruction);
        if(statistical)check(YK::estimateStatisticalWorkspace(system,statistical_workspace),"Cannot estimate statistical workspace for requested modules");
        if (wfbp) {
            system.reconstruction.wfbp.filter_cutoff = static_cast<float>(config.reconstruction.analytic.wfbp_cutoff);
            system.reconstruction.wfbp.filter_apodization = static_cast<float>(config.reconstruction.analytic.wfbp_apodization);
        }
        if(statistical){const auto& s=config.reconstruction.statistical;
            std::cout<<" optimizer="<<s.optimizer.type<<" iterations="<<s.optimizer.iterations<<" regularization="<<s.regularization.plugin
                <<" strength="<<s.regularization.strength<<" shape="<<s.regularization.shape<<" adjoint=approximate";
        }else if (iterative) {
            std::cout << " iterations=" << config.reconstruction.iterative.iterations
                << " relaxation=" << config.reconstruction.iterative.relaxation
                << " subsets=" << config.reconstruction.iterative.subsets;
        }
        std::cout << "reconstruction=" << pipeline_name
            << " filter=" << (wfbp ? "freect-ramp" : config.reconstruction.analytic.filter);
        if (wfbp) std::cout << " cutoff=" << config.reconstruction.analytic.wfbp_cutoff
            << " apodization=" << config.reconstruction.analytic.wfbp_apodization
            << " full_input_bytes=" << projection_elements * sizeof(float)
            << " chunk_views=not-applicable";
        std::cout << std::endl;
        if(progress)progress({"Initializing reconstruction",0,0});
        const bool batch_capable=iterative && config.geometry.kind!=GeometryKind::CylCbct && config.geometry.kind!=GeometryKind::CylHelical &&
            (statistical||pipeline_name=="sart"||pipeline_name=="sirt"||pipeline_name=="ossart"||pipeline_name=="tigre_sart"||pipeline_name=="tigre_os_sart"||pipeline_name=="tigre_sirt"||pipeline_name=="tigre_sart_tv"||pipeline_name=="tigre_os_sart_tv");
        if(iterative && (statistical || pipeline==YK::EPipeline::TigreGradient || pipeline==YK::EPipeline::SART || pipeline==YK::EPipeline::SIRT || pipeline==YK::EPipeline::OSSART)){
            const bool single=pipeline_name=="sart"||pipeline_name=="tigre_sart"||pipeline_name=="tigre_sart_tv";
            const bool tigre_os=pipeline_name=="tigre_os_sart"||pipeline_name=="tigre_os_sart_tv";
            const int exact_block=tigre_os?std::min(g.views,system.reconstruction.tigre.block_size):0;
            const bool all=statistical||pipeline_name=="tigre_sirt"||pipeline_name=="sirt"||(tigre_os?exact_block>=g.views:(!single&&config.reconstruction.iterative.subsets==1));
            const size_t block=single?1:all?size_t(YK::detail::iterativeBatchViews(pixels,g.views)):tigre_os?size_t(exact_block):size_t((g.views+std::max(1,config.reconstruction.iterative.subsets)-1)/std::max(1,config.reconstruction.iterative.subsets));
            const bool tv=pipeline_name=="tigre_sart_tv"||pipeline_name=="tigre_os_sart_tv";
            const auto& regularization=config.reconstruction.iterative.regularization;
            const size_t tv_buffers=regularization.plugin=="smoothed_tv"?1:
                regularization.plugin=="huber_tv_prox"&&regularization.strength>0?2:0;
            const size_t workspace=statistical?statistical_workspace.solverBytes()+statistical_workspace.caller_volume_bytes+statistical_workspace.weight_staging_bytes:
                ((size_t((pipeline_name=="ossart"||pipeline_name=="sirt")?5:3)+tv_buffers)*volume_elements+4*block*pixels+
                size_t(g.reconstruction_volume_x)*g.reconstruction_volume_y+
                (tv?6*volume_elements+7*size_t(YK::detail::iterativeBatchViews(pixels,g.views))*pixels:0))*sizeof(float)+size_t(32)*1024*1024;
            const size_t resident=workspace+projection_elements*sizeof(float);
            size_t free=0,total=0;check(cudaMemGetInfo(&free,&total)==cudaSuccess,"Cannot query reconstruction memory");
            const size_t reserve=std::max(size_t(512)*1024*1024,total/5);
            const size_t available=free>reserve?free-reserve:0;
            std::string mode="auto";if(const char* requested=std::getenv("YK_RECON_PROJECTION_INPUT"))mode=requested;
            check(mode=="auto"||mode=="resident"||mode=="batched","Invalid YK_RECON_PROJECTION_INPUT");
            projection_input_mode=mode;planned_block=block;
            batched_projection=batch_capable&&(mode=="batched"||(mode=="auto"&&resident>available));
            check(mode!="batched"||batch_capable,"This algorithm does not support batched projection input");
            const size_t frame_bytes=pixels*sizeof(float);
            const size_t batch_views=std::min(size_t(g.views),std::min(std::max(size_t(64),block),size_t(256)*1024*1024/frame_bytes));
            check(!batched_projection||batch_views>=block,"投影批缓冲无法容纳一个算法子集，请增加 OS-SART 子集数量");
            planned_batch_bytes=batch_views*frame_bytes;
            const size_t required=batched_projection?workspace+planned_batch_bytes:resident;
            check(YK::detail::fitsReconstructionBudget(required,"projection input and reconstruction"),
                "重建显存预算不足（已预留显示空间），请降低投影尺寸、重建网格或子集大小");
        }
        session = YK::ReconstructionSessionFactory::create();
        check(session != nullptr, "无法创建 YKCBCT DLL Session");
        bool initialized=false;{YK::diagnostics::TimingSpan watch("reconstruction.initialize",pipeline_name);initialized=session->initialize(system);}
        check(initialized, std::string("YKCBCT 初始化失败: ") + session->lastErrorMessage());
        if(batch_capable){
            size_t free=0,total=0;check(cudaMemGetInfo(&free,&total)==cudaSuccess,"Cannot query initialized reconstruction memory");
            const size_t reserve=std::max(size_t(512)*1024*1024,total/5);
            const size_t remaining=(2*volume_elements+planned_block*pixels+((pipeline_name=="tigre_sart_tv"||pipeline_name=="tigre_os_sart_tv")?volume_elements+6*size_t(YK::detail::iterativeBatchViews(pixels,g.views))*pixels:0))*sizeof(float)+size_t(32)*1024*1024;
            const size_t available=free>reserve?free-reserve:0;
            if(projection_input_mode=="auto"&&!batched_projection&&projection_elements*sizeof(float)+remaining>available)
                batched_projection=true;
            const size_t input_bytes=batched_projection?planned_batch_bytes:projection_elements*sizeof(float);
            check(input_bytes+remaining<=available,"初始化后显存余量不足，请降低重建网格或子集大小");
            std::cout<<"reconstruction_projection mode="<<(batched_projection?"batched":"resident")<<" input_buffer_bytes="<<input_bytes
                <<" free_bytes="<<free<<" reserve_bytes="<<reserve<<std::endl;
        }


        std::vector<float> volume(volume_elements, 0.f);
        if (fdk && !cyl_fdk && !nonstandard_fdk) {
            std::vector<float> chunk;
            const int chunk_views = std::min(g.views,
                std::max(1, config.reconstruction.analytic.chunk_views));
            if(progress)progress({"FDK views",0,g.views});
            for (int offset = 0; offset < g.views; offset += chunk_views) {
                const int count = std::min(chunk_views, g.views - offset);
                chunk.resize(static_cast<std::size_t>(count) * pixels);
                timedRead(input,reinterpret_cast<char*>(chunk.data()),
                    static_cast<std::streamsize>(chunk.size() * sizeof(float)));
                check(static_cast<bool>(input), "读取投影分块失败");
                YK::SExecutionRequest request{};
                request.projection = {chunk.data(), YK::EMemoryLocation::Host,
                    chunk.size()};
                request.volume = {volume.data(), YK::EMemoryLocation::Host,
                    volume.size()};
                request.view_offset = offset;
                request.view_count = count;
                request.clear_output = offset == 0;
                {YK::diagnostics::TimingSpan watch("reconstruction.execute",pipeline_name);
                check(session->execute(request), std::string("YKCBCT FDK 执行失败: ") + session->lastErrorMessage());
                if(YK::diagnostics::timingEnabled()&&cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("reconstruction profiling synchronization failed");
                }
                if(progress)progress({"FDK views",offset+count,g.views});
            }
        } else if (wfbp || cyl_fdk || nonstandard_fdk) {
            // 全量 Host 输入交给 DLL；显存由 Session 的统一分配器管理。
            std::vector<float> projection(projection_elements);
            timedRead(input,reinterpret_cast<char*>(projection.data()),
                static_cast<std::streamsize>(projection.size() * sizeof(float)));
            check(static_cast<bool>(input), "读取完整 wFBP 投影失败");
            YK::SExecutionRequest request{};
            request.projection = {projection.data(), YK::EMemoryLocation::Host,
                projection.size()};
            request.volume = {volume.data(), YK::EMemoryLocation::Host,
                volume.size()};
            request.view_count = g.views;
            if(progress)progress({"Full-volume reconstruction",0,1});
            const bool reconstructed = session->execute(request);
            if(reconstructed && progress)progress({"Full-volume reconstruction",1,1});
            check(reconstructed, std::string("YKCBCT " + pipeline_name + " 执行失败: ") + session->lastErrorMessage());
        } else {
            std::vector<float> projection;
            if(!batched_projection){
                projection.resize(projection_elements);
                timedRead(input,reinterpret_cast<char*>(projection.data()),static_cast<std::streamsize>(projection.size()*sizeof(float)));
                check(static_cast<bool>(input),"读取完整迭代重建投影失败");
            }
            struct FileProjectionReader {
                std::ifstream& input;std::filesystem::path path;size_t pixels;int views;
                std::filesystem::file_time_type modified;uintmax_t bytes;
                static bool read(void* data,const int* indices,int count,float* output,size_t elements){
                    auto& self=*static_cast<FileProjectionReader*>(data);
                    check(elements==size_t(count)*self.pixels,"Projection reader element mismatch");
                    check(std::filesystem::file_size(self.path)==self.bytes&&std::filesystem::last_write_time(self.path)==self.modified,"Projection file changed during batched reconstruction");
                    for(int i=0;i<count;++i){
                        check(indices[i]>=0&&indices[i]<self.views,"Projection reader index out of range");
                        self.input.clear();self.input.seekg(std::streamoff(size_t(indices[i])*self.pixels*sizeof(float)));
                        timedRead(self.input,reinterpret_cast<char*>(output+size_t(i)*self.pixels),std::streamsize(self.pixels*sizeof(float)));
                        check(bool(self.input),"读取投影批次失败，view="+std::to_string(indices[i]));
                    }return true;
                }
            } reader{input,reconstruction_input,pixels,g.views,std::filesystem::last_write_time(reconstruction_input),std::filesystem::file_size(reconstruction_input)};
            float* d_projection = nullptr;
            float* d_volume = nullptr;
            auto cudaCheck = [](cudaError_t e, const char* what) {
                if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
            };
            try {
                if(!batched_projection)cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_projection), projection.size()*sizeof(float)),"分配迭代投影显存失败");
                cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_volume), volume.size() * sizeof(float)), "分配迭代体显存失败");
                if(!batched_projection)cudaCheck(cudaMemcpy(d_projection,projection.data(),projection.size()*sizeof(float),cudaMemcpyHostToDevice),"上传迭代投影失败");
                cudaCheck(cudaMemset(d_volume, 0, volume.size() * sizeof(float)), "清零迭代体失败");
                YK::SExecutionRequest request{};
                request.projection = {d_projection, YK::EMemoryLocation::Device, projection.size()};
                request.volume = {d_volume, YK::EMemoryLocation::Device, volume.size()};
                if(batched_projection){request.projection_read=&FileProjectionReader::read;request.projection_read_user_data=&reader;}
                request.view_count = g.views;
                request.iteration_count = statistical?config.reconstruction.statistical.optimizer.iterations:config.reconstruction.iterative.iterations;
                struct WeightReader {
                    std::ifstream input;std::filesystem::path path;size_t pixels;int views;
                    uintmax_t bytes;std::filesystem::file_time_type modified;
                    static bool read(void* data,const int* ids,int count,float* out,size_t elements){
                        auto& r=*static_cast<WeightReader*>(data);
                        check(elements==size_t(count)*r.pixels,"Weight reader shape mismatch");
                        check(std::filesystem::file_size(r.path)==r.bytes&&std::filesystem::last_write_time(r.path)==r.modified,"Weight file changed during reconstruction");
                        for(int i=0;i<count;++i){check(ids[i]>=0&&ids[i]<r.views,"Weight reader index");r.input.clear();
                            r.input.seekg(std::streamoff(size_t(ids[i])*r.pixels*sizeof(float)));timedRead(r.input,reinterpret_cast<char*>(out+size_t(i)*r.pixels),std::streamsize(r.pixels*sizeof(float)));
                            check(bool(r.input),"Cannot read statistical weight batch");}return true;
                    }
                } weights;
                if(statistical&&!config.reconstruction.statistical.data_fidelity.weights_file.empty()){
                    weights.path=config.reconstruction.statistical.data_fidelity.weights_file;weights.pixels=pixels;weights.views=g.views;
                    weights.bytes=std::filesystem::file_size(weights.path);weights.modified=std::filesystem::last_write_time(weights.path);
                    check(weights.bytes==projection_elements*sizeof(float),"Statistical weight file has incorrect shape");
                    weights.input.open(weights.path,std::ios::binary);check(bool(weights.input),"Cannot open statistical weight file");
                    request.projection_weights_read=&WeightReader::read;request.projection_weights_read_user_data=&weights;
                }
                if(progress){
                    progress({"Iterative updates",0,request.iteration_count});
                    request.progress_callback=[](void* data,int complete,int total){auto& callback=*static_cast<TaskProgressCallback*>(data);callback({"Iterative updates",complete,total});};
                    request.progress_user_data=&progress;
                }
                {YK::diagnostics::TimingSpan watch("reconstruction.execute",pipeline_name);
                check(session->execute(request), std::string("YKCBCT 迭代重建执行失败: ") + session->lastErrorMessage());
                if(YK::diagnostics::timingEnabled()&&cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("reconstruction profiling synchronization failed");
                }
                cudaCheck(cudaMemcpy(volume.data(), d_volume, volume.size() * sizeof(float), cudaMemcpyDeviceToHost), "下载迭代重建体失败");
            } catch (...) {
                if (d_projection) cudaFree(d_projection);
                if (d_volume) cudaFree(d_volume);
                throw;
            }
            cudaFree(d_projection);
            cudaFree(d_volume);
        }
        if(progress)progress({"Saving reconstruction",0,1});
        if (const auto parent = volume_file.parent_path(); !parent.empty())
            std::filesystem::create_directories(parent);
        std::ofstream output(volume_file, std::ios::binary);
        check(static_cast<bool>(output), "无法创建重建体输出: " + volume_file.string());
        timedWrite(output,reinterpret_cast<const char*>(volume.data()),
            static_cast<std::streamsize>(volume.size() * sizeof(float)));
        check(static_cast<bool>(output), "写入重建体失败");
        if (config.reconstruction.save_slices)
            writeSlices(slice_prefix, volume, g.reconstruction_volume_x,
                g.reconstruction_volume_y, g.reconstruction_volume_z);
        YK::ReconstructionSessionFactory::destroy(session);
        session = nullptr;
        if(progress)progress({"Completed",1,1});
        diagnostic = pipeline_name +
            " 重建完成，输出体素=" + std::to_string(volume.size());
        return true;
    } catch (const std::exception& e) {
        if (session) YK::ReconstructionSessionFactory::destroy(session);
        diagnostic = e.what();
        return false;
    }
}

} // namespace yk::spectral
