#include "ui/ReconstructionViewerPanel.hpp"
#include <iostream>

using namespace yk::visualization;
using yk::spectral::ui::ReconstructionViewerPanel;
namespace {
void require(bool ok, const char* text)
{
    if (!ok)
        throw std::runtime_error(text);
}
}
int main()
{
    auto* context = ImGui::CreateContext();
    try {
        auto& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.DisplaySize = { 1600, 1000 };
        io.DeltaTime = 1.f / 60;
        unsigned char* font_pixels = nullptr;
        int width = 0, height = 0;
        io.Fonts->GetTexDataAsRGBA32(&font_pixels, &width, &height);
        LiveView view({ 512, 15, true, false, {} });
        FloatVolume volume;
        volume.dimensions = { 31, 31, 31 };
        volume.spacing = { 1, 1, 1 };
        volume.values.assign(31 * 31 * 31, .02f);
        ReconstructionViewerPanel panel;
        panel.reset(volume);
        view.setReconstruction(std::move(volume));
        auto frame = [&] {ImGui::NewFrame();panel.imageInteractions(view,true);ImGui::Render();
            if(panel.slices_changed){panel.slices_changed=false;panel.mpr.indices=panel.recon_slices;view.setReconstructionMpr(panel.mpr);} };
        auto center = view.reconstructionCrosshair(0);
        const auto before_indices = panel.recon_slices;
        io.AddMousePosEvent(float(center[0] + 64), float(center[1]));
        frame();
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
        frame();
        require(panel.rotation_panel == 0, "reference handle did not enter rotation mode");
        io.AddMousePosEvent(float(center[0] + 45), float(center[1] - 45));
        frame();
        require(!mprReferenceIsOrthogonal(panel.mpr), "pointer drag did not rotate MPR");
        require(std::abs(panel.mpr.rotation_deg[2] - 45) < .01, "angle editor did not follow reference drag");
        auto editor = panel.mpr;
        setMprRotation(editor, panel.mpr.rotation_deg);
        for (int i = 0; i < 9; ++i)
            require(std::abs(editor.reference_axes[i] - panel.mpr.reference_axes[i]) < 1e-9, "angle editor and reference geometry disagree");
        require(panel.recon_slices == before_indices, "rotation drag moved linked cursor");
        auto lines = view.reconstructionReferenceLines(0);
        bool diagonal = false;
        for (const auto& line : lines)
            if (std::abs(line.direction[0]) > .3 && std::abs(line.direction[1]) > .3)
                diagonal = true;
        require(diagonal, "reference line did not follow rotated geometry");
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
        frame();
        require(panel.rotation_panel == -1, "release did not end rotation");
        const auto axes = panel.mpr.reference_axes;
        center = view.reconstructionCrosshair(0);
        io.AddMousePosEvent(float(center[0]), float(center[1]));
        frame();
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
        frame();
        require(panel.rotation_panel == -1, "center entered rotation mode");
        io.AddMousePosEvent(float(center[0] + 9), float(center[1] + 9));
        frame();
        require(panel.recon_slices != before_indices, "center drag did not move cursor");
        require(panel.mpr.reference_axes == axes, "center drag rotated reference axes");
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
        frame();
        ImGui::DestroyContext(context);
        std::cout << "MPR drag rotation, release and center movement passed\n";
        return 0;
    } catch (const std::exception& e) {
        ImGui::DestroyContext(context);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
