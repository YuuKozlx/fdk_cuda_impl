#include "input/config/ReconstructionGeometry.hpp"
#include "input/config/SimConfig.hpp"
#include "input/materials/XcomAttenuationProvider.hpp"
#include "recon/bhc/BoneBhcRouting.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
namespace yk::spectral {
void normalizeConfig(SimulationConfig& c)
{
    auto& sc = c.reconstruction.preprocessing.scatter;
    if (sc.enabled) {
        if (sc.air_projection_file.empty())
            sc.air_projection_file = c.projection.air_output_file;
        if (sc.model_file.empty())
            sc.model_file = c.projection.scatter.model_file;
    }
    if (c.runsReconstruction() && c.reconstruction.input_projection_file.empty())
        c.reconstruction.input_projection_file = c.projection.output_file;
}
namespace {
    void validateConfiguration(const SimulationConfig& c, bool check_runtime_files)
    {
        const auto& hu = c.reconstruction.postprocessing.hu_correction;
        if (c.runsReconstruction() && hu.enabled) {
            if (hu.calibration_file.empty())
                throw std::runtime_error("HU correction requires a saved calibration_file");
            if (check_runtime_files && !std::filesystem::is_regular_file(hu.calibration_file))
                throw std::runtime_error("HU calibration file does not exist: " + hu.calibration_file.string());
        }
        const auto& response = c.projection.detector_response;
        if (c.runsProjection() && response.enabled) {
            for (const auto* layer : { &response.protective_layer, &response.impurity_layer, &response.scintillator_layer }) {
                if (!layer->enabled)
                    continue;
                if (!findBuiltinMaterial(layer->preset))
                    throw std::runtime_error("unknown detector layer preset: " + layer->preset);
                if (!std::isfinite(layer->thickness_mm) || (layer->thickness_mm <= 0 && layer->thickness_map_file.empty()))
                    throw std::runtime_error("detector layer requires positive thickness_mm or thickness_map_file");
                if (check_runtime_files && !layer->thickness_map_file.empty() && !std::filesystem::is_regular_file(layer->thickness_map_file))
                    throw std::runtime_error("detector layer thickness map does not exist");
            }
        }
        const auto validate_kernel = [](const std::vector<double>& values, const char* key) {
            for (double v : values)
                if (!std::isfinite(v) || v < 0)
                    throw std::runtime_error(std::string(key) + " must contain finite non-negative numbers");
            if (values.empty() || values.size() % 2 == 0 || values.size() > 31)
                throw std::runtime_error(std::string(key) + " must have an odd length between 1 and 31");
            double sum = 0.0;
            for (double value : values)
                sum += value;
            if (std::abs(sum - 1.0) > 1e-6)
                throw std::runtime_error(std::string(key) + " weights must sum to 1");
        };
        for (const auto* cross : { &c.projection.detector_postprocess.optical_crosstalk, &c.projection.detector_postprocess.electronic_crosstalk }) {
            validate_kernel(cross->kernel_u, "kernel_u");
            validate_kernel(cross->kernel_v, "kernel_v");
        }
        {
            const auto& value = c.projection.detector_postprocess.afterglow;
            if (value.initialization != "zero" && value.initialization != "steady_state")
                throw std::runtime_error("afterglow.initialization must be zero or steady_state");
            if (value.enabled) {
                if (!std::isfinite(value.p) || !std::isfinite(value.frame_time_ms) || !std::isfinite(value.exposure_time_ms) || value.p < 0 || value.p >= 1 || value.frame_time_ms <= 0 || value.exposure_time_ms <= 0 || value.exposure_time_ms > value.frame_time_ms || value.weights.empty() || value.weights.size() != value.time_constants_ms.size())
                    throw std::runtime_error("afterglow requires 0 < exposure_time_ms <= frame_time_ms and equally sized weights/time_constants_ms");
                double sum = 0.0;
                for (size_t i = 0; i < value.weights.size(); ++i) {
                    if (!std::isfinite(value.weights[i]) || !std::isfinite(value.time_constants_ms[i]) || value.weights[i] < 0 || value.time_constants_ms[i] <= 0)
                        throw std::runtime_error("afterglow weights must be non-negative and time constants positive");
                    sum += value.weights[i];
                }
                if (std::abs(sum - 1.0) > 1e-6)
                    throw std::runtime_error("afterglow weights must sum to 1");
            }
        }
        {
            const auto& value = c.projection.detector_postprocess.das;
            if (value.enabled && (!std::isfinite(value.gain_electrons_per_keV) || value.gain_electrons_per_keV <= 0 || !std::isfinite(value.offset_electrons) || !std::isfinite(value.electronic_noise_std_electrons) || value.electronic_noise_std_electrons < 0 || !std::isfinite(value.saturation_electrons) || value.saturation_electrons < 0 || !std::isfinite(value.adc_lsb_electrons) || value.adc_lsb_electrons < 0))
                throw std::runtime_error("invalid detector DAS parameters");
        }
        const auto available = [check_runtime_files](const std::filesystem::path& p) {
            return !check_runtime_files || std::filesystem::is_regular_file(p);
        };
        {
            const auto& v = c.visualization;
            if (!std::isfinite(v.volume_opacity_scale) || v.volume_opacity_scale < 0 || v.volume_opacity_scale > 1)
                throw std::runtime_error("visualization.volume_opacity_scale must be 0..1");
            if (v.texture_max_size < 1 || v.texture_max_size > 8192 || !std::isfinite(v.max_fps) || v.max_fps <= 0 || v.max_fps > 120 || (v.projection != "signal" && v.projection != "primary" && v.projection != "log"))
                throw std::runtime_error("invalid visualization options");
            if (v.enabled && !c.runsProjection())
                throw std::runtime_error("visualization requires a projection workflow");
            if (v.interactive && (!v.enabled || c.workflow.mode != WorkflowMode::Project || v.offscreen || !v.hold_open))
                throw std::runtime_error("interactive visualization requires enabled=true, project workflow, offscreen=false and hold_open=true");
        }
        if (c.runsProjection()) {
            const auto& sp = c.projection.spectrum;
            validateSpectrumSource(sp);
            if (sp.source == "generated" && !c.projection.spectrum_file.empty())
                throw std::runtime_error("generated spectrum conflicts with projection.spectrum_file");
        }
        if (c.runsProjection()) {

            if (!std::isfinite(c.projection.mAs_per_view) || c.projection.mAs_per_view <= 0.0)
                throw std::runtime_error("projection.mAs_per_view must be positive");
        }
        if (c.runsProjection()) {

            if (c.projection.scatter.kernel.mode != "all" && c.projection.scatter.kernel.mode != "segmented")
                throw std::runtime_error("projection.scatter.kernel.mode must be all or segmented");
            if (c.projection.scatter.kernel.mode == "segmented" && c.projection.scatter.kernel.segments < 2)
                throw std::runtime_error("projection.scatter.kernel.segments must be at least 2");

            if (c.projection.scatter.enabled) {
                if (!available(c.projection.scatter.model_file))
                    throw std::runtime_error("scatter.model_file must point to an existing kernel model");
                if (c.geometry.kind == GeometryKind::CylCbct || c.geometry.kind == GeometryKind::CylHelical)
                    throw std::runtime_error("kernel scatter currently supports flat detectors only");
            }
        }
        {
            for (const auto& f : c.reconstruction.preprocessing.filters) {
                if ((f.operation != "mean" && f.operation != "gaussian" && f.operation != "guided") || f.radius_pixels < 1 || f.radius_pixels > 64 || !std::isfinite(f.sigma_pixels) || f.sigma_pixels <= 0 || !std::isfinite(f.epsilon) || f.epsilon <= 0)
                    throw std::runtime_error("Invalid preprocessing projection filter");
            }
            if (c.reconstruction.preprocessing.filters.size() > 16)
                throw std::runtime_error("At most 16 preprocessing filters are supported");
        }
        {
            const auto& sc = c.reconstruction.preprocessing.scatter;
            if (sc.solver.type != "fixed" && sc.solver.type != "adaptive_experimental")
                throw std::runtime_error("scatter.solver.type must be fixed or adaptive_experimental");
            const auto& adaptive = sc.solver.adaptive;
            if (!std::isfinite(adaptive.damping_decay) || adaptive.damping_decay <= 0 || adaptive.damping_decay > 1)
                throw std::runtime_error("adaptive damping_decay must be in (0,1]");
            if (sc.guide.enabled && (!std::isfinite(sc.guide.sigma_u_pixels) || sc.guide.sigma_u_pixels <= 0 || sc.guide.sigma_u_pixels > 32 || !std::isfinite(sc.guide.sigma_v_pixels) || sc.guide.sigma_v_pixels <= 0 || sc.guide.sigma_v_pixels > 32))
                throw std::runtime_error("scatter guide sigma must be in (0,32] pixels");
            if (adaptive.iterations <= 0 || !std::isfinite(adaptive.initial_damping) || adaptive.initial_damping <= 0.0 || adaptive.initial_damping > 1.0 || !std::isfinite(adaptive.min_damping) || adaptive.min_damping <= 0.0 || adaptive.min_damping > adaptive.initial_damping)
                throw std::runtime_error("invalid experimental adaptive scatter solver parameters");
            const auto& guard = adaptive.transmission_guard;
            if (!std::isfinite(guard.low_quantile) || guard.low_quantile <= 0 || guard.low_quantile > 1 || !std::isfinite(guard.risk_threshold) || guard.risk_threshold <= 0 || guard.risk_threshold >= 1 || !std::isfinite(guard.positivity_margin) || guard.positivity_margin <= 0 || guard.positivity_margin >= 1)
                throw std::runtime_error("invalid adaptive scatter transmission guard");
        }
        if (c.reconstruction.type == "statistical") {
            const auto& cfg = c.reconstruction.statistical;
            if (cfg.forward_projector != "auto" && cfg.forward_projector != "joseph" && cfg.forward_projector != "siddon" && cfg.forward_projector != "footprint")
                throw std::runtime_error("Statistical forward_projector supports auto/joseph/siddon/footprint");
            if (cfg.back_projector != "auto" && cfg.back_projector != "fdk_pseudo_matched" && cfg.back_projector != "siddon_ray" && cfg.back_projector != "joseph" && cfg.back_projector != "joseph_v3" && cfg.back_projector != "footprint")
                throw std::runtime_error("Statistical back_projector is unsupported");
            const auto& d = cfg.data_fidelity;
            const auto& reg = cfg.regularization;
            const auto& opt = cfg.optimizer;
            if (d.type != "weighted_least_squares" || d.require_verified_adjoint)
                throw std::runtime_error("Statistical data fidelity unavailable or adjoint not verified");
            if (opt.type != "regularized_algebraic" && opt.type != "gradient_descent" && opt.type != "algebraic_gradient" && opt.type != "raw_gradient_descent")
                throw std::runtime_error("Statistical optimizer not implemented: FISTA is reserved");
            if (reg.plugin != "none" && reg.plugin != "smoothed_tv" && reg.plugin != "huber_tv" && reg.plugin != "huber_tv_prox" && reg.plugin != "tikhonov" && !(reg.plugin == "tv" && (opt.type == "regularized_algebraic" || opt.type == "algebraic_gradient")))
                throw std::runtime_error("Gradient Descent requires a differentiable regularizer; standard TV requires Prox");
            if (!std::isfinite(d.scale) || d.scale <= 0 || !std::isfinite(reg.strength) || reg.strength < 0 || ((reg.plugin != "none" && reg.plugin != "tikhonov" && !(reg.plugin == "tv" && (opt.type == "regularized_algebraic" || opt.type == "algebraic_gradient"))) && (!std::isfinite(reg.shape) || reg.shape <= 0)) || (reg.dimensions != 2 && reg.dimensions != 3) || (reg.plugin == "none" && reg.strength > 0) || opt.iterations <= 0 || !std::isfinite(opt.step_size) || opt.step_size < 0 || opt.max_backtracking < 0 || !std::isfinite(opt.step_shrink) || opt.step_shrink <= 0 || opt.step_shrink >= 1 || !std::isfinite(opt.relative_tolerance) || opt.relative_tolerance < 0 || opt.minimum_iterations <= 0)
                throw std::runtime_error("Invalid statistical reconstruction parameters");
            if (check_runtime_files && !d.weights_file.empty()) {
                const auto& g = c.geometry.parameters;
                const uintmax_t expected = uintmax_t(g.views) * g.detector_u * g.detector_v * sizeof(float);
                if (!std::filesystem::exists(d.weights_file) || std::filesystem::file_size(d.weights_file) != expected)
                    throw std::runtime_error("Statistical weights_file must be float32 with the projection shape");
            }
        }
        if (c.reconstruction.type == "analytic") {
            const auto& rc = c.reconstruction.analytic;
            if (rc.pipeline != "fdk" && rc.pipeline != "wfbp")
                throw std::runtime_error("analytic.pipeline 仅支持 fdk/wfbp");
            if (rc.filter != "ramlak" && rc.filter != "ram-lak" && rc.filter != "shepp-logan" && rc.filter != "cosine" && rc.filter != "hann" && rc.filter != "hamming" && rc.filter != "blackman" && rc.filter != "butterworth" && rc.filter != "kaiser" && rc.filter != "tukey")
                throw std::runtime_error("未知 analytic.filter: " + rc.filter);
            if (rc.chunk_views <= 0 || !std::isfinite(rc.cutoff) || rc.cutoff <= 0 || rc.cutoff > .5 || !std::isfinite(rc.gain) || rc.gain <= 0 || !std::isfinite(rc.butterworth_order) || rc.butterworth_order <= 0 || !std::isfinite(rc.kaiser_beta) || rc.kaiser_beta < 0 || !std::isfinite(rc.tukey_alpha) || rc.tukey_alpha < 0 || rc.tukey_alpha > 1)
                throw std::runtime_error("Invalid shared analytic filter settings");
        }
        if (c.reconstruction.type == "iterative") {
            const auto& rc = c.reconstruction.iterative;
            if (rc.forward_projector != "auto" && rc.forward_projector != "joseph" && rc.forward_projector != "siddon" && rc.forward_projector != "footprint")
                throw std::runtime_error("forward_projector supports auto/joseph/siddon/footprint");
            if (rc.back_projector != "auto" && rc.back_projector != "fdk_pseudo_matched" && rc.back_projector != "siddon_ray" && rc.back_projector != "joseph" && rc.back_projector != "joseph_v3" && rc.back_projector != "footprint")
                throw std::runtime_error("back_projector is unsupported");
            {
                const auto& tv = rc.regularization;
                if (tv.plugin != "none" && tv.plugin != "huber_tv_prox" && tv.plugin != "smoothed_tv")
                    throw std::runtime_error("Unknown regularization plugin");
                if (tv.plugin != "none") {
                    if (rc.algorithm != "sart" && rc.algorithm != "sirt" && rc.algorithm != "ossart")
                        throw std::runtime_error("TV plugins require internal sart/sirt/ossart");
                    if (c.geometry.kind == GeometryKind::CylCbct || c.geometry.kind == GeometryKind::CylHelical)
                        throw std::runtime_error("TV plugins require flat detector");
                    if (!std::isfinite(tv.strength) || tv.strength < 0 || tv.inner_iterations <= 0 || (tv.dimensions != 2 && tv.dimensions != 3) || !std::isfinite(tv.huber_delta) || tv.huber_delta <= 0 || !std::isfinite(tv.step_size) || tv.step_size < 0 || !std::isfinite(tv.epsilon) || tv.epsilon <= 0 || !std::isfinite(tv.strength_reduction) || tv.strength_reduction <= 0)
                        throw std::runtime_error("Invalid TV plugin parameters");
                }
            }
            if (rc.algorithm != "tigre_sart" && rc.algorithm != "tigre_sirt" && rc.algorithm != "tigre_os_sart" && rc.algorithm != "tigre_sart_tv" && rc.algorithm != "tigre_os_sart_tv" && rc.algorithm != "sart" && rc.algorithm != "sirt" && rc.algorithm != "ossart" && rc.algorithm != "cgls")
                throw std::runtime_error("iterative.algorithm 不支持该值");
            const bool uses_tv = rc.algorithm == "tigre_sart_tv" || rc.algorithm == "tigre_os_sart_tv";
            if (uses_tv && rc.tv_alpha <= 0)
                throw std::runtime_error("TIGRE TV algorithm requires tv_alpha > 0");
            const bool uses_subsets = rc.algorithm == "ossart" || rc.algorithm == "tigre_os_sart" || rc.algorithm == "tigre_os_sart_tv";
            if (!uses_subsets && rc.subsets != 1)
                throw std::runtime_error("该 iterative.algorithm 不使用 subsets，配置值必须为 1");
            if (rc.iterations <= 0 || !std::isfinite(rc.relaxation) || rc.relaxation <= 0 || rc.subsets <= 0 || (uses_tv && (rc.tv_iterations <= 0 || !std::isfinite(rc.tv_alpha) || rc.tv_alpha < 0 || !std::isfinite(rc.tv_alpha_reduction) || rc.tv_alpha_reduction <= 0)) || !std::isfinite(rc.maximum_update_ratio) || rc.maximum_update_ratio <= 0)
                throw std::runtime_error("iterative 参数必须为正数");
        }
        if (c.runsReconstruction() || c.reconstruction.preprocessing.enabled || c.reconstruction.bone_hardening.enabled) {

            auto bone_validation = c;
            if (c.reconstruction.geometry_override.enabled)
                bone_validation.geometry.parameters = resolveReconstructionGeometry(c);
            const auto bone_route = bone_bhc::resolveRouting(bone_validation);
            if (c.reconstruction.bone_hardening.enabled && bone_route.use_lut && !available(c.reconstruction.bone_hardening.calibration_file))
                throw std::runtime_error("Bone hardening calibration file does not exist");
            auto& pp = c.reconstruction.preprocessing;
            if (pp.enabled) {
                if (!pp.beam_hardening.enabled && !pp.scatter.enabled && pp.filters.empty())
                    throw std::runtime_error("reconstruction.preprocessing must enable filters, beam_hardening or scatter");
                const auto& bh = pp.beam_hardening;
                if (bh.enabled && !bh.calibration_file.empty() && !available(bh.calibration_file)) throw std::runtime_error("Water BHC calibration file does not exist");
                if (bh.enabled && bh.calibration_file.empty() && (bh.order < 1 || bh.order > 16 || (!bh.fit_from_water_plate && bh.coefficients.size() != static_cast<std::size_t>(bh.order + 1)) || !std::isfinite(bh.clamp_min) || !std::isfinite(bh.clamp_max) || bh.clamp_min < 0.0 || bh.clamp_max <= bh.clamp_min))
                    throw std::runtime_error("invalid beam hardening polynomial configuration");
                if (bh.enabled && bh.calibration_file.empty() && bh.fit_from_water_plate && ((!(bh.spectrum_file.empty() && c.projection.spectrum.source == "generated") && !available(bh.spectrum_file.empty() ? c.projection.spectrum_file : bh.spectrum_file)) || !std::isfinite(bh.water_plate_max_mm) || bh.water_plate_max_mm <= 0.0 || bh.water_plate_samples < bh.order + 2 || !std::isfinite(bh.reference_energy_keV) || bh.reference_energy_keV < 0.0))
                    throw std::runtime_error("invalid water-plate beam hardening calibration configuration");
                auto& sc = pp.scatter;
                // GUI setup can carry defaults from the forward model, but execution remains strict.
                if (sc.enabled) {
                    if (sc.air_projection_file.empty())
                        throw std::runtime_error("scatter correction requires an air projection file (set Correction air or projection.air_output_file)");
                    if (sc.model_file.empty())
                        throw std::runtime_error("scatter correction requires a model file (set Correction model)");
                    const bool planned_air = c.runsProjection() && sc.air_projection_file.lexically_normal() == c.projection.air_output_file.lexically_normal();
                    if (!available(sc.air_projection_file) && !planned_air)
                        throw std::runtime_error("scatter correction air projection file does not exist: " + sc.air_projection_file.string());
                    if (!available(sc.model_file))
                        throw std::runtime_error("scatter correction model file does not exist: " + sc.model_file.string());
                    if (sc.correction.iterations <= 0)
                        throw std::runtime_error("scatter correction iterations must be positive");
                    if (!std::isfinite(sc.correction.damping) || sc.correction.damping <= 0 || sc.correction.damping > 1)
                        throw std::runtime_error("scatter correction damping must be in (0,1]");
                    if (!std::isfinite(sc.correction.scatter_scale) || sc.correction.scatter_scale < 0)
                        throw std::runtime_error("scatter correction scale must be finite and nonnegative");
                    if (sc.kernel.mode != "all" && sc.kernel.mode != "segmented")
                        throw std::runtime_error("scatter correction kernel mode must be all or segmented");
                    if (sc.kernel.mode == "segmented" && sc.kernel.segments < 2)
                        throw std::runtime_error("scatter correction kernel segments must be at least 2");
                    if (!std::isfinite(sc.correction.object_mask_threshold) || sc.correction.object_mask_threshold < 0)
                        throw std::runtime_error("scatter correction object mask threshold must be finite and nonnegative");
                }
            }
        }
        {

            if (!std::isfinite(c.geometry.parameters.phantom_rotation_x_rad) || !std::isfinite(c.geometry.parameters.phantom_rotation_y_rad) || !std::isfinite(c.geometry.parameters.phantom_rotation_z_rad))
                throw std::runtime_error("geometry.phantom_rotation_*_rad 必须为有限数值");
        }
        {

            if (c.runsProjection() && c.geometry.kind != GeometryKind::FlatCbct && c.geometry.kind != GeometryKind::FlatHelical && c.geometry.kind != GeometryKind::CylCbct && c.geometry.kind != GeometryKind::CylHelical && c.geometry.kind != GeometryKind::PlanarSourceOrbit && c.geometry.kind != GeometryKind::NikonTilted)
                throw std::runtime_error("随机多能谱前投不支持该几何类型");
        }
        {

            if (c.runsProjection() && c.projection.materials.size() > 64)
                throw std::runtime_error("随机多能谱前投当前最多支持64种材料");
        }
        if (c.runsProjection()) {
            const auto& spot = c.projection.focal_spot;
            if (!std::isfinite(spot.epsilon) || spot.epsilon < 0.0 || spot.epsilon > 1.0)
                throw std::runtime_error("projection.focal_spot.epsilon must be in [0, 1]");
            if ((spot.enabled || spot.defocus_enabled) && ((!std::isfinite(spot.normal_size_u_mm) || spot.normal_size_u_mm < 0) || (!std::isfinite(spot.normal_size_v_mm) || spot.normal_size_v_mm < 0) || (!std::isfinite(spot.defocus_size_u_mm) || spot.defocus_size_u_mm < 0) || (!std::isfinite(spot.defocus_size_v_mm) || spot.defocus_size_v_mm < 0) || (spot.enabled && (!spot.defocus_enabled || spot.epsilon < 1.0) && (spot.normal_size_u_mm <= 0 || spot.normal_size_v_mm <= 0)) || (spot.defocus_enabled && spot.epsilon > 0.0 && (spot.defocus_size_u_mm <= 0 || spot.defocus_size_v_mm <= 0))))
                throw std::runtime_error("矩形焦点尺寸必须为正");
            if (spot.enabled && spot.defocus_enabled && spot.epsilon > 0.0 && (spot.defocus_size_u_mm < spot.normal_size_u_mm || spot.defocus_size_v_mm < spot.normal_size_v_mm))
                throw std::runtime_error("散焦矩形必须包含正常焦点矩形，才能在外圈分层采样");
        }
        if (c.runsProjection()) {

            if (c.projection.batching.views < 1 || c.projection.batching.views > 64 || c.projection.batching.max_buffer_mib < 1 || c.projection.batching.max_buffer_mib > 256)
                throw std::runtime_error("projection.batching: views must be 1..64, max_buffer_mib must be 1..256");
            if (c.projection.sampling.samples_per_pixel <= 0 || c.projection.sampling.samples_per_pixel > 4096)
                throw std::runtime_error("projection.sampling.samples_per_pixel must be in 1..4096");
            const bool planar = c.geometry.kind == GeometryKind::PlanarSourceOrbit;
            const bool tilted = c.geometry.kind == GeometryKind::NikonTilted;
            if ((planar || tilted) && (!(c.geometry.parameters.scan_range_rad > 0.0) || !std::isfinite(c.geometry.parameters.scan_range_rad)))
                throw std::runtime_error("non-standard geometry requires positive scan_range_rad");
            if (planar && (!(c.geometry.parameters.planar_detector_y_mm > 0.0) || c.geometry.parameters.reconstruction_detector_u <= 0 || c.geometry.parameters.reconstruction_detector_v <= 0 || (c.geometry.parameters.planar_resample_mode != "in_plane_rotated" && c.geometry.parameters.planar_resample_mode != "perspective_orthogonal")))
                throw std::runtime_error("invalid planar_source_orbit geometry parameters");
            if ((planar || tilted) && c.projection.detector_response.enabled)
                throw std::runtime_error("non-standard trajectories do not yet support detector response");
        }

        const auto& g = c.geometry.parameters;
        if (c.schema_version != 2)
            throw std::runtime_error("schema_version must be 2");
        if (g.views <= 0 || g.views_per_turn <= 0 || g.detector_u <= 0 || g.detector_v <= 0)
            throw std::runtime_error("View count and detector dimensions must be positive");
        if (g.rotation_direction != 1 && g.rotation_direction != -1)
            throw std::runtime_error("Rotation direction must be +1 or -1");
        for (double value : { g.sid_mm, g.sdd_mm, g.pixel_u_mm, g.pixel_v_mm })
            if (!std::isfinite(value) || value <= 0)
                throw std::runtime_error("Scan distances and pixel spacing must be finite and positive");
        if (g.sdd_mm <= g.sid_mm)
            throw std::runtime_error("SDD must be greater than SID");
        if (c.runsProjection()) {
            if (g.volume_x <= 0 || g.volume_y <= 0 || g.volume_z <= 0)
                throw std::runtime_error("Volume dimensions must be positive");
            if (c.projection.output_file.empty())
                throw std::runtime_error("Projection output path is required");
            if (c.projection.materials.empty())
                throw std::runtime_error("At least one material is required");
            std::array<bool, 256> labels {};
            for (const auto& m : c.projection.materials) {
                if (labels[m.label])
                    throw std::runtime_error("Duplicate material label");
                labels[m.label] = true;
                if (m.formula.empty() == m.preset.empty() || !std::isfinite(m.density_g_cm3) || m.density_g_cm3 <= 0)
                    throw std::runtime_error("Material needs formula or preset and positive density");
                if (!m.preset.empty() && !findBuiltinMaterial(m.preset))
                    throw std::runtime_error("Unknown material preset");
            }
        }
        if (c.runsReconstruction()) {
            if (c.reconstruction.input_projection_file.empty())
                throw std::runtime_error("Reconstruction input path is required");
            if (c.reconstruction.type != "analytic" && c.reconstruction.type != "iterative" && c.reconstruction.type != "statistical")
                throw std::runtime_error("Unknown reconstruction type");
            const bool helical = c.geometry.kind == GeometryKind::FlatHelical || c.geometry.kind == GeometryKind::CylHelical;
            if (c.reconstruction.type == "analytic" && (c.reconstruction.analytic.pipeline == "wfbp") != helical)
                throw std::runtime_error("FDK requires circular geometry; WFBP requires helical geometry");
        }
        resolveReconstructionGeometry(c);
    }
}
void validateConfig(const SimulationConfig& c) { validateConfiguration(c, false); }
void validateRuntimeInputs(const SimulationConfig& c) { validateConfiguration(c, true); }
}
