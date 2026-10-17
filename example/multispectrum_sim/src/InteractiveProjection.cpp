#include "MultiSpecFpStage.hpp"
#include "InteractiveProjection.hpp"
#ifdef YK_SPECTRAL_IMGUI
#include "ImGuiProjection.hpp"
#endif
#include "SpectralProjectionPipeline.hpp"
#include "primary/CudaProjectionBackend.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace yk::spectral {
namespace {
    constexpr double degrees = 57.2957795130823208768;
    std::vector<yk::visualization::EditableParameter> parameters(const GeometryConfig& g)
    {
        using P = yk::visualization::EditableParameter;
        return { P { "Object X (mm)", g.phantom_offset_x_mm, -500, 500 }, P { "Object Y (mm)", g.phantom_offset_y_mm, -500, 500 }, P { "Object Z (mm)", g.phantom_offset_z_mm, -500, 500 },
            P { "Object Rx (deg)", g.phantom_rotation_x_rad * degrees, -180, 180 }, P { "Object Ry (deg)", g.phantom_rotation_y_rad * degrees, -180, 180 }, P { "Object Rz (deg)", g.phantom_rotation_z_rad * degrees, -180, 180 },
            P { "Source X (mm)", g.source_offset_x_mm, -100, 100 }, P { "Source Y (mm)", g.source_offset_y_mm, -100, 100 }, P { "Source Z (mm)", g.source_offset_z_mm, -100, 100 },
            P { "Detector U (mm)", g.offset_u_mm, -100, 100 }, P { "Detector N (mm)", g.offset_n_mm, -100, 100 }, P { "Detector V (mm)", g.offset_v_mm, -100, 100 },
            P { "Detector tilt U (deg)", g.tilt_u_rad * degrees, -90, 90 }, P { "Detector tilt V (deg)", g.tilt_v_rad * degrees, -90, 90 }, P { "Detector tilt N (deg)", g.tilt_n_rad * degrees, -90, 90 },
            P { "SID (mm)", g.sid_mm, 10, 3000 }, P { "SDD (mm)", g.sdd_mm, 10, 5000 },
            P { "Columns", double(g.detector_u), 2, 4096, true }, P { "Rows", double(g.detector_v), 2, 4096, true },
            P { "Pixel U (mm)", g.pixel_u_mm, .01, 5 }, P { "Pixel V (mm)", g.pixel_v_mm, .01, 5 } };
    }
    void applyValues(GeometryConfig& g, const std::vector<double>& v)
    {
        if (v.size() != 21)
            throw std::runtime_error("invalid parameter panel");
        g.phantom_offset_x_mm = v[0];
        g.phantom_offset_y_mm = v[1];
        g.phantom_offset_z_mm = v[2];
        g.phantom_rotation_x_rad = v[3] / degrees;
        g.phantom_rotation_y_rad = v[4] / degrees;
        g.phantom_rotation_z_rad = v[5] / degrees;
        g.source_offset_x_mm = v[6];
        g.source_offset_y_mm = v[7];
        g.source_offset_z_mm = v[8];
        g.offset_u_mm = v[9];
        g.offset_n_mm = v[10];
        g.offset_v_mm = v[11];
        g.tilt_u_rad = v[12] / degrees;
        g.tilt_v_rad = v[13] / degrees;
        g.tilt_n_rad = v[14] / degrees;
        g.sid_mm = v[15];
        g.sdd_mm = v[16];
        g.detector_u = int(std::lround(v[17]));
        g.detector_v = int(std::lround(v[18]));
        g.pixel_u_mm = v[19];
        g.pixel_v_mm = v[20];
    }
}
bool runInteractiveProjection(const SimulationConfig& initial, const std::vector<std::uint8_t>& labels,
    const SpectralTransmissionModel& model, const std::filesystem::path& output, std::string& error)
{
#ifdef YK_SPECTRAL_IMGUI
    return runImGuiProjection(initial,output,error,&labels);
#else
    try {
        auto current = initial;
        current.projection.sampling.samples_per_pixel = 1;
        const auto& v = current.visualization;
        yk::visualization::LiveView view({ v.texture_max_size, v.max_fps, false, true, v.snapshot_directory.string() });
        view.setParameters(parameters(current.geometry.parameters));
        SpectralProjectionPipeline pipeline(primary_detail::makeCudaProjectionSourceRegistry().create("projection.source.multi_spec_fp"));
        pipeline.setVisualization(&view);
        bool success = false;
        for (;;) {
            view.setEditingEnabled(false);
            view.setStatus("Calculating all views (1 ray/pixel). Parameters and scan slider are locked until completion.");
            success = pipeline.run(current, labels, model, output, error);
            if (!view.isOpen())
                return success;
            view.setEditingEnabled(true);
            if (success) {
                view.setStatus("Ready: all views cached. Drag scan slider to browse; adjust parameters and press Apply / Enter to recalculate.");
                std::ofstream used(output.string() + ".vtk-parameters.txt");
                used.precision(17);
                for (const auto& parameter : parameters(current.geometry.parameters))
                    used << parameter.label << " = " << parameter.value << '\n';
                used << "samples_per_pixel = 1\n";
            } else
                view.setStatus("Calculation failed: " + error);
            bool rerun = false;
            while (view.isOpen()) {
                view.finish();
                if (!view.isOpen())
                    return success;
                if (!view.takeRecomputeRequest())
                    continue;
                auto trial = current;
                applyValues(trial.geometry.parameters, view.parameterValues());
                if (trial.geometry.parameters.sdd_mm <= trial.geometry.parameters.sid_mm) {
                    view.setStatus("Parameters not applied: SDD must be greater than SID.");
                    continue;
                }
                current = std::move(trial);
                rerun = true;
                break;
            }
            if (!rerun)
                return success;
        }
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
#endif
}
}
