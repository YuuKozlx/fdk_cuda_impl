#include "../../../visualization/RelativeGeometry.hpp"
#include <iostream>
using namespace yk::visualization;
void require(bool v, const char* s)
{
    if (!v)
        throw std::runtime_error(s);
}
void near(const Point& a, const Point& b)
{
    for (int i = 0; i < 3; ++i)
        require(std::abs(a[i] - b[i]) < 1e-9, "coordinate mismatch");
}
int main()
{
    try {
        FrameGeometry first;
        first.source = { -709, 0, -100 };
        first.detector_center = { 434, 0, -100 };
        first.detector_axes = { Point { 0, 1, 0 }, Point { 0, 0, 1 }, Point { 1, 0, 0 } };
        first.detector = { { 434, -100, -110 }, { 434, 100, -110 }, { 434, -100, -90 }, { 434, 100, -90 } };
        first.object_pose = { 1, 0, 0, 12, 0, .8, -.6, 7, 0, .6, .8, -3, 0, 0, 0, 1 };
        const auto same = relativeGeometry(first, first);
        near(posePoint(same.object_pose, { 2, 3, 4 }), posePoint(first.object_pose, { 2, 3, 4 }));
        // A quarter turn plus 40 mm helical translation, with object pose unchanged.
        const Pose motion = { 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 40, 0, 0, 0, 1 };
        auto next = first;
        next.source = posePoint(motion, first.source);
        next.detector_center = posePoint(motion, first.detector_center);
        for (auto& p : next.detector)
            p = posePoint(motion, p);
        for (auto& a : next.detector_axes)
            a = posePoint(motion, a, true);
        next.view = 90;
        const auto relative = relativeGeometry(first, next);
        near(relative.source, first.source);
        near(relative.detector_center, first.detector_center);
        for (size_t i = 0; i < first.detector.size(); ++i)
            near(relative.detector[i], first.detector[i]);
        near(posePoint(relative.object_pose, {}), { 7, -12, -43 });
        require(relative.view == 90, "view lost");
        require(rigidGantryMatches(first, relative), "rigid helix rejected");
        const auto transform = relativeWorldPose(first, next);
        // Source/object/detector all receive one rigid transform, preserving ray incidence.
        near(posePoint(relative.object_pose, { 2, 3, 4 }), posePoint(transform, posePoint(next.object_pose, { 2, 3, 4 })));
        auto circle = next;
        circle.source[2] -= 40;
        circle.detector_center[2] -= 40;
        for (auto& p : circle.detector)
            p[2] -= 40;
        const auto circular = relativeGeometry(first, circle);
        require(rigidGantryMatches(first, circular), "circular gantry rejected");
        near(posePoint(circular.object_pose, {}), { 7, -12, -3 });
        auto shifted = next;
        shifted.source[0] += 1;
        require(!rigidGantryMatches(first, relativeGeometry(first, shifted)), "nonrigid source motion accepted");
        near(next.source, posePoint(motion, first.source)); // Original computation geometry remains unchanged.
        std::cout << "Relative circular/helical geometry, object pose and nonrigid guard passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
