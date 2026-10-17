#include "Mpr.hpp"
#include <iostream>
using namespace yk::visualization;
namespace {
void require(bool ok, const char* message)
{
    if (!ok)
        throw std::runtime_error(message);
}
double linear(const std::array<double, 3>& p) { return 2 * p[0] + 3 * p[1] + 5 * p[2] + 7; }
}
int main()
{
    try {
        FloatVolume volume;
        volume.dimensions = { 11, 9, 7 };
        volume.spacing = { .5, 1, 1.5 };
        volume.origin = { -10, 20, -30 };
        volume.direction = { 0, -1, 0, 1, 0, 0, 0, 0, 1 };
        for (int z = 0; z < 7; ++z)
            for (int y = 0; y < 9; ++y)
                for (int x = 0; x < 11; ++x)
                    volume.values.push_back(float(linear({ x * .5, double(y), z * 1.5 })));
        validateFloatVolume(volume);
        MprReslicer reslicer;
        reslicer.setInput(volume);
        MprSettings state;
        state.indices = { 5, 4, 3 };
        for (int panel = 0; panel < 3; ++panel) {
            auto plane = makeMprPlane(volume, state, panel);
            auto slice = reslicer.slice(plane, MprInterpolation::Linear);
            const auto expected = extractFloatSlice(volume, panel == 0 ? 2 : panel == 1 ? 1
                                                                                        : 0,
                state.indices[panel == 0 ? 2 : panel == 1 ? 1
                                                          : 0]);
            require(slice.values.size() == expected.values.size(), "orthogonal slice size");
            for (size_t i = 0; i < slice.values.size(); ++i)
                require(std::abs(slice.values[i] - expected.values[i]) < 1e-4, "orthogonal orientation or anisotropic spacing");
        }
        for (const auto rotation : { std::array<double, 3> { 0, 0, 0 }, { 0, 90, 0 }, { 23, -37, 61 } }) {
            state.rotation_deg = rotation;
            state.offset_mm = .27;
            auto p = makeMprPlane(volume, state, 3);
            auto s = reslicer.slice(p, MprInterpolation::Linear);
            int checked = 0, outside = 0;
            for (int y = 0; y < s.height; ++y)
                for (int x = 0; x < s.width; ++x) {
                    std::array<double, 3> point {};
                    bool inside = true;
                    for (int a = 0; a < 3; ++a) {
                        point[a] = p.center[a] + p.u[a] * (p.start[0] + x * p.spacing[0]) + p.v[a] * (p.start[1] + y * p.spacing[1]);
                        inside = inside && point[a] >= 1e-3 && point[a] < (volume.dimensions[a] - 1) * volume.spacing[a] - 1e-3;
                    }
                    const auto value = s.values[size_t(y) * s.width + x];
                    if (inside) {
                        require(std::isfinite(value) && std::abs(value - linear(point)) < .003, "oblique trilinear interpolation or rotation order");
                        ++checked;
                    } else if (!std::isfinite(value))
                        ++outside;
                }
            require(checked > 10, "oblique plane did not intersect volume");
            if (rotation[0] != 0)
                require(outside > 0, "oblique background should be masked");
            auto nearest = reslicer.slice(p, MprInterpolation::Nearest);
            auto cubic = reslicer.slice(p, MprInterpolation::Cubic);
            require(nearest.values.size() == s.values.size() && cubic.values.size() == s.values.size(), "interpolation modes");
        }
        const auto world = mprWorldPosition(volume, state.indices);
        // Each drag must turn the other two planes, preserve orthogonality and
        // keep the active image basis (including its roll) stable.
        MprSettings linked;
        linked.indices = { 5, 4, 3 };
        for (int turn = 0; turn < 120; ++turn) {
            const int active = turn % 3;
            const auto before = makeMprPlane(volume, linked, active);
            rotateMprReference(linked, active, turn == 0 ? .6 : .017);
            const auto after = makeMprPlane(volume, linked, active);
            for (int a = 0; a < 3; ++a) {
                require(std::abs(before.u[a] - after.u[a]) < 1e-10 && std::abs(before.v[a] - after.v[a]) < 1e-10, "active image rolled during reference drag");
                require(before.center[a] == after.center[a], "reference rotation moved cursor");
            }
            for (int i = 0; i < 3; ++i)
                for (int j = i + 1; j < 3; ++j) {
                    const auto p = makeMprPlane(volume, linked, i), q = makeMprPlane(volume, linked, j);
                    double dot = 0;
                    for (int a = 0; a < 3; ++a)
                        dot += p.normal[a] * q.normal[a];
                    require(std::abs(dot) < 1e-10, "linked planes lost orthogonality");
                }
            if (turn == 0)
                require(before.width == after.width && before.height == after.height && before.spacing == after.spacing, "active image sampling changed on first drag");
        }
        require(!mprReferenceIsOrthogonal(linked), "reference rotation was ignored");
        auto editor = linked;
        setMprRotation(editor, linked.rotation_deg);
        for (int i = 0; i < 9; ++i)
            require(std::abs(editor.reference_axes[i] - linked.reference_axes[i]) < 1e-9, "drag angles do not represent reference rotation");
        for (const auto angles : { std::array<double, 3> { 23, 37, -61 }, { 10, 90, 40 }, { 10, -90, 40 }, { 175, 89.999, -178 } }) {
            MprSettings trial;
            setMprRotation(trial, angles);
            const auto original = trial.reference_axes;
            rotateMprReference(trial, 0, 0);
            auto roundtrip = trial;
            setMprRotation(roundtrip, trial.rotation_deg);
            for (int i = 0; i < 9; ++i)
                require(std::abs(original[i] - roundtrip.reference_axes[i]) < 1e-8, "angle roundtrip at gimbal lock");
            const auto axial = makeMprPlane(volume, trial, 0), oblique = makeMprPlane(volume, trial, 3);
            for (int i = 0; i < 3; ++i)
                require(std::abs(axial.normal[i] - oblique.normal[i]) < 1e-8, "fourth view orientation not synchronized");
        }
        for (int panel = 0; panel < 3; ++panel) {
            const auto p = makeMprPlane(volume, linked, panel);
            const auto slice = reslicer.slice(p, MprInterpolation::Linear);
            int checked = 0;
            for (int y = 0; y < slice.height; ++y)
                for (int x = 0; x < slice.width; ++x) {
                    std::array<double, 3> point {};
                    bool inside = true;
                    for (int a = 0; a < 3; ++a) {
                        point[a] = p.center[a] + p.u[a] * (p.start[0] + x * p.spacing[0]) + p.v[a] * (p.start[1] + y * p.spacing[1]);
                        inside = inside && point[a] > .01 && point[a] < (volume.dimensions[a] - 1) * volume.spacing[a] - .01;
                    }
                    if (inside) {
                        require(std::abs(slice.values[size_t(y) * slice.width + x] - linear(point)) < .003, "linked plane sampled wrong coordinates");
                        ++checked;
                    }
                }
            require(checked > 10, "linked plane missing volume");
        }
        require(world == std::array<double, 3> { -14, 22.5, -25.5 }, "world origin/direction mapping");
        state.rotation_deg = {};
        state.offset_mm = 0;
        const auto plane = makeMprPlane(volume, state, 0);
        require(mprPick(volume, plane, 2, 6) == std::array<int, 3> { 2, 6, 3 }, "linked cursor mapping");
        state.offset_mm = 1000;
        auto outside = reslicer.slice(makeMprPlane(volume, state, 3), MprInterpolation::Linear);
        for (float v : outside.values)
            require(!std::isfinite(v), "outside plane background");
        reslicer.clear();
        bool rejected = false;
        try {
            reslicer.slice(plane, MprInterpolation::Linear);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "use after clear");
        reslicer.setInput(volume);
        state.offset_mm = 0;
        state.rotation_deg[0] = std::numeric_limits<double>::quiet_NaN();
        rejected = false;
        try {
            makeMprPlane(volume, state, 3);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "nonfinite angle accepted");
        require(attenuationToHu(.02, .02) == 0 && attenuationToHu(0, .02) == -1000, "HU reference");
        std::cout << "MPR orthogonal, oblique, anisotropy, coordinates and interpolation passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
