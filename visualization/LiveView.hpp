#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace yk::visualization {
using Point = std::array<double, 3>;
using Pose = std::array<double, 16>; // Row-major, local point -> world point.
inline Pose identityPose() { return { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }; }
struct FrameGeometry {
    int view = 0;
    Point source {};
    // Two edge rows, U increasing. Covers pixel boundaries, not pixel centers.
    std::vector<Point> detector;
    Point detector_center {};
    std::array<Point, 3> detector_axes {};
    Pose object_pose = identityPose();
};
struct MaterialColor {
    std::uint8_t label = 0;
    std::string name;
    std::array<double, 4> rgba { 0.8, 0.8, 0.8, 0.15 };
};
struct LabelVolume {
    std::array<int, 3> dimensions {};
    Point spacing {};
    std::vector<std::uint8_t> labels; // [z][y][x], x fastest, never resampled.
    std::vector<MaterialColor> materials;
};
struct LiveViewOptions {
    int texture_max_size = 512;
    double max_fps = 15;
    bool offscreen = false;
    bool hold_open = true;
    std::string snapshot_directory; // Optional final PNG/VTK diagnostic export.
};
struct EditableParameter {
    std::string label;
    double value = 0, minimum = 0, maximum = 1;
    bool integral = false;
};
// Call on the window-owning thread. Inputs are copied; no retained producer pointers.
// Camera interaction never changes physical scene geometry.
class LiveView {
public:
    explicit LiveView(const LiveViewOptions&);
    ~LiveView();
    LiveView(const LiveView&) = delete;
    LiveView& operator=(const LiveView&) = delete;
    void setVolume(const LabelVolume&);
    void invalidateResults();
    void setTrajectory(const std::vector<Point>& sources, const std::vector<Point>& detector_centers);
    void setParameters(const std::vector<EditableParameter>&);
    std::vector<double> parameterValues() const;
    bool setParameter(std::size_t index, double value);
    void setEditingEnabled(bool enabled);
    void setStatus(const std::string&);
    void requestRecompute();
    bool takeRecomputeRequest();
    bool isOpen() const;
    bool wantsFrame(); // Pumps window events; close disables display only.
    void update(const FrameGeometry&, int width, int height,
        const std::vector<float>& projection, const std::string& description, bool force_render = false);
    void updateProjections(const FrameGeometry&, int width, int height,
        const std::vector<float>& texture, const std::vector<float>& energy,
        const std::vector<float>& attenuation, const std::string& description, bool force_render = false);
    bool selectFrame(int view); // Select a computed frame, including its cached display texture.
    int displayedFrame() const;
    std::size_t cachedFrameCount() const;
    void finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace yk::visualization
