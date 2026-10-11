#include "Scene.hpp"
#include "CLI11/CLI11.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vtkActor.h>
#include <vtkAxesActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCaptionActor2D.h>
#include <vtkCellArray.h>
#include <vtkColorTransferFunction.h>
#include <vtkCompositeDataSet.h>
#include <vtkCubeSource.h>
#include <vtkGPUVolumeRayCastMapper.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkCellData.h>
#include <vtkFieldData.h>
#include <vtkMatrix4x4.h>
#include <vtkMatrix3x3.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkNew.h>
#include <vtkPiecewiseFunction.h>
#include <vtkPNGWriter.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSliderRepresentation2D.h>
#include <vtkSliderWidget.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkStringArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>
#include <vtkWindowToImageFilter.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLMultiBlockDataWriter.h>
#include <vtkXMLPolyDataWriter.h>

namespace fs = std::filesystem;
using Viewer::Point;

vtkSmartPointer<vtkPolyData> lines(const std::vector<Point>& p,
    const std::vector<std::vector<vtkIdType>>& cells)
{
    vtkNew<vtkPoints> points;
    points->SetDataTypeToDouble();
    for (const auto& v : p) points->InsertNextPoint(v.data());
    vtkNew<vtkCellArray> edges;
    for (const auto& cell : cells) edges->InsertNextCell(vtkIdType(cell.size()), cell.data());
    auto data = vtkSmartPointer<vtkPolyData>::New();
    data->SetPoints(points); data->SetLines(edges);
    return data;
}

vtkSmartPointer<vtkActor> meshActor(vtkPolyData* data, Point color,
    double opacity = 1.)
{
    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputData(data);
    auto actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(color.data());
    actor->GetProperty()->SetOpacity(opacity);
    return actor;
}

class View {
public:
    explicit View(const Viewer::Options& options) : options_(options), scene_(options) {
        renderer_->SetBackground(0.035, 0.055, 0.095);
        renderer_->SetBackground2(0.11, 0.16, 0.22);
        renderer_->GradientBackgroundOn();
        window_->AddRenderer(renderer_); window_->SetSize(1200, 800);
        window_->SetWindowName("YKCBCT - VTK geometry and volume viewer");
        window_->SetMultiSamples(0);
        if (options.offscreen || !options.screenshot.empty()) window_->SetOffScreenRendering(1);
        interactor_->SetRenderWindow(window_);
        vtkNew<vtkInteractorStyleTrackballCamera> style;
        interactor_->SetInteractorStyle(style);

        box_->SetCenter(options.center.data());
        box_->SetXLength(options.size[0]*options.spacing[0]);
        box_->SetYLength(options.size[1]*options.spacing[1]);
        box_->SetZLength(options.size[2]*options.spacing[2]); box_->Update();
        box_actor_ = meshActor(box_->GetOutput(), {0.62, 0.75, 0.88}, 0.75);
        box_actor_->GetProperty()->SetRepresentationToWireframe();
        box_actor_->GetProperty()->SetLineWidth(1.5);
        renderer_->AddActor(box_actor_);
        image_ = makeImage();
        vtkNew<vtkGPUVolumeRayCastMapper> volume_mapper;
        volume_mapper->SetInputData(image_); volume_mapper->SetBlendModeToComposite();
        vtkNew<vtkColorTransferFunction> colors;
        colors->AddRGBPoint(0., 0.04, 0.07, 0.1);
        const auto range = image_->GetScalarRange();
        const double high = std::max(range[1], 1e-8);
        colors->AddRGBPoint(high*0.25, 0.23, 0.45, 0.72);
        colors->AddRGBPoint(high*0.5, 0.62, 0.80, 0.89);
        colors->AddRGBPoint(high, 1., 0.93, 0.74);
        vtkNew<vtkPiecewiseFunction> alpha;
        alpha->AddPoint(range[0], 0.); alpha->AddPoint(high*0.2, 0.);
        alpha->AddPoint(high*0.45, 0.035); alpha->AddPoint(high, 0.25);
        vtkNew<vtkVolumeProperty> property;
        property->SetColor(colors); property->SetScalarOpacity(alpha);
        property->SetInterpolationTypeToLinear(); property->ShadeOn();
        property->SetAmbient(0.3); property->SetDiffuse(0.7);
        volume_->SetMapper(volume_mapper); volume_->SetProperty(property);
        volume_->SetVisibility(!options.box_only); renderer_->AddVolume(volume_);

        source_->SetRadius(options.sid*0.025);
        source_->SetThetaResolution(24); source_->SetPhiResolution(24);
        vtkNew<vtkPolyDataMapper> source_mapper;
        source_mapper->SetInputConnection(source_->GetOutputPort());
        source_actor_->SetMapper(source_mapper);
        source_actor_->GetProperty()->SetColor(1., 0.65, 0.16);
        renderer_->AddActor(source_actor_);
        detector_actor_ = meshActor(detector_, {0.08, 0.82, 0.77}, 0.55);
        detector_actor_->GetProperty()->EdgeVisibilityOn();
        detector_actor_->GetProperty()->SetEdgeColor(0.02, 0.3, 0.32);
        renderer_->AddActor(detector_actor_);
        rays_actor_ = meshActor(rays_, {0.22, 0.69, 0.88}, 0.3);
        renderer_->AddActor(rays_actor_);
        trajectory_actor_ = meshActor(trajectory_, {1., 0.65, 0.16}, 0.55);
        trajectory_actor_->GetProperty()->SetLineWidth(2.);
        renderer_->AddActor(trajectory_actor_);
        auto* text_property = title_->GetTextProperty();
        text_property->SetFontSize(25); text_property->SetColor(0.91, 0.95, 1.);
        title_->SetDisplayPosition(30, 752); renderer_->AddViewProp(title_);
        status_->GetTextProperty()->SetFontSize(15);
        status_->GetTextProperty()->SetColor(0.65, 0.76, 0.88);
        status_->SetDisplayPosition(30, 718); renderer_->AddViewProp(status_);
        help_->SetInput("1 Flat circular   2 Cyl circular   3 Flat helical   4 Cyl helical   5 Source orbit   6 Nikon tilted\nLeft/Right: frame   M: relative mode   V: volume   T: trail   L: local axes   R: camera reset   E: export\nSource axes: U/R/V (R = beam)   Detector axes: U/V/N (N = inward normal)   Units: mm");
        help_->GetTextProperty()->SetFontSize(14); help_->GetTextProperty()->SetColor(0.68, 0.80, 0.88);
        help_->SetDisplayPosition(30, 110); renderer_->AddViewProp(help_);
        axes_->SetTotalLength(25., 25., 25.);
        axes_->SetPosition(options.center[0], options.center[1], options.center[2]); renderer_->AddActor(axes_);
        for (auto* caption : {axes_->GetXAxisCaptionActor2D(), axes_->GetYAxisCaptionActor2D(),
                axes_->GetZAxisCaptionActor2D()}) {
            caption->GetTextActor()->SetTextScaleModeToNone();
            caption->GetCaptionTextProperty()->SetFontSize(14);
        }
        configureLocalAxes(source_axes_, {"S-U", "S-R", "S-V"});
        configureLocalAxes(detector_axes_, {"D-U", "D-V", "D-N"});

        slider_rep_->SetMinimumValue(0); slider_rep_->SetMaximumValue(options.views-1);
        slider_rep_->SetValue(options.frame); slider_rep_->SetTitleText("Projection frame");
        slider_rep_->SetLabelHeight(0.018); slider_rep_->SetTitleHeight(0.024);
        slider_rep_->ShowSliderLabelOff(); // Current frame is already shown in the status row.
        slider_rep_->GetPoint1Coordinate()->SetCoordinateSystemToNormalizedDisplay();
        slider_rep_->GetPoint1Coordinate()->SetValue(0.15, 0.10);
        slider_rep_->GetPoint2Coordinate()->SetCoordinateSystemToNormalizedDisplay();
        slider_rep_->GetPoint2Coordinate()->SetValue(0.85, 0.10);
        slider_rep_->GetTitleProperty()->SetColor(0.9, 0.95, 1.);
        slider_rep_->GetLabelProperty()->SetColor(0.9, 0.95, 1.);
        slider_->SetInteractor(interactor_); slider_->SetRepresentation(slider_rep_);
        slider_->SetAnimationModeToJump();
        slider_callback_->SetClientData(this);
        slider_callback_->SetCallback([](vtkObject*, unsigned long, void* data, void*) {
            auto& view = *static_cast<View*>(data);
            view.setFrame(int(std::lround(view.slider_rep_->GetValue())));
            view.window_->Render();
        });
        slider_->AddObserver(vtkCommand::InteractionEvent, slider_callback_);
        key_callback_->SetClientData(this);
        key_callback_->SetCallback([](vtkObject*, unsigned long, void* data, void*) {
            auto& view = *static_cast<View*>(data);
            const std::string key = view.interactor_->GetKeySym();
            try { view.key(key); } catch (const std::exception& e) { view.status_->SetInput(e.what()); }
            view.window_->Render();
        });
        interactor_->AddObserver(vtkCommand::KeyPressEvent, key_callback_);
        rebuildTrajectory(); setFrame(options.frame); resetCamera();
    }

    void run() {
        interactor_->Initialize(); slider_->EnabledOn(); window_->Render();
        if (options_.self_check) checkInteraction();
        if (!options_.export_dir.empty()) exportScene(options_.export_dir);
        if (!options_.screenshot.empty()) {
            fs::path p(options_.screenshot);
            if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
            vtkNew<vtkWindowToImageFilter> capture;
            capture->SetInput(window_); capture->ReadFrontBufferOff(); capture->Update();
            vtkNew<vtkPNGWriter> png; png->SetFileName(p.string().c_str());
            png->SetInputConnection(capture->GetOutputPort()); png->Write();
            if (png->GetErrorCode() || !fs::exists(p)) throw std::runtime_error("Screenshot write failed");
            std::cout << "Screenshot: " << fs::absolute(p) << '\n';
        }
        if (!options_.offscreen && options_.screenshot.empty()) interactor_->Start();
        slider_->EnabledOff(); window_->Finalize();
    }

private:
    vtkSmartPointer<vtkImageData> makeImage() {
        auto image = vtkSmartPointer<vtkImageData>::New();
        const auto& o = options_;
        image->SetDimensions(o.size.data()); image->SetSpacing(o.spacing.data());
        const auto origin = scene_.volume.origin(); image->SetOrigin(origin.x, origin.y, origin.z);
        image->AllocateScalars(VTK_FLOAT, 1);
        auto* buffer = static_cast<float*>(image->GetScalarPointer());
        const size_t count = size_t(o.size[0])*o.size[1]*o.size[2];
        if (!o.raw.empty()) {
            std::ifstream input(o.raw, std::ios::binary);
            if (!input) throw std::runtime_error("Cannot open RAW volume");
            input.read(reinterpret_cast<char*>(buffer), std::streamsize(count*sizeof(float)));
            if (size_t(input.gcount()) != count*sizeof(float) || input.peek() != EOF)
                throw std::runtime_error("RAW size does not match --size (float32, x fastest)");
        } else {
            for (int z = 0; z < o.size[2]; ++z)
                for (int y = 0; y < o.size[1]; ++y)
                    for (int x = 0; x < o.size[0]; ++x) {
                        const double px = (x-(o.size[0]-1)*0.5)/(o.size[0]*0.37);
                        const double py = (y-(o.size[1]-1)*0.5)/(o.size[1]*0.37);
                        const double pz = (z-(o.size[2]-1)*0.5)/(o.size[2]*0.37);
                        float value = px*px+py*py+pz*pz < 1. ? 0.02f : 0.f;
                        if ((px-0.3)*(px-0.3)+py*py+pz*pz < 0.10) value = 0.05f;
                        buffer[(size_t(z)*o.size[1]+y)*o.size[0]+x] = value;
                    }
        }
        for (size_t i = 0; i < count; ++i)
            if (!std::isfinite(buffer[i])) throw std::runtime_error("Volume contains non-finite values");
        return image;
    }

    void rebuildTrajectory() {
        std::vector<Point> points;
        std::vector<vtkIdType> ids;
        for (int i = 0; i < options_.views; ++i) {
            points.push_back(options_.gantry_fixed
                ? scene_.gantryFixedTransform(i).position(Viewer::point(scene_.volume.center))
                : scene_.source(i));
            ids.push_back(i);
        }
        trajectory_->ShallowCopy(lines(points, {ids}));
    }

    void setFrame(int frame) {
        frame_ = std::clamp(frame, 0, options_.views-1);
        slider_rep_->SetValue(frame_);
        display_pose_ = options_.gantry_fixed ? scene_.gantryFixedTransform(frame_) : Viewer::DisplayTransform{};
        vtkNew<vtkMatrix4x4> matrix;
        matrix->Identity();
        for (int row = 0; row < 3; ++row) {
            matrix->SetElement(row, 3, display_pose_.translation[row]);
            for (int col = 0; col < 3; ++col) matrix->SetElement(row, col, display_pose_.rows[row][col]);
        }
        display_transform_->SetMatrix(matrix);
        box_actor_->SetUserTransform(display_transform_); volume_->SetUserTransform(display_transform_);
        axes_->SetUserTransform(display_transform_);
        axes_->SetXAxisLabelText(options_.gantry_fixed ? "O-X" : "X");
        axes_->SetYAxisLabelText(options_.gantry_fixed ? "O-Y" : "Y");
        axes_->SetZAxisLabelText(options_.gantry_fixed ? "O-Z" : "Z");
        const auto s = display_pose_.position(scene_.source(frame_));
        source_->SetCenter(s.data()); source_->Update();
        updateLocalAxes(source_axes_, source_axes_data_, display_pose_.frame(scene_.sourceFrame(frame_)), {"S-U", "S-R", "S-V"});
        updateLocalAxes(detector_axes_, detector_axes_data_, display_pose_.frame(scene_.detectorFrame(frame_)), {"D-U", "D-V", "D-N"});
        const int strips = scene_.cyl.empty() ? 1 : 64;
        vtkNew<vtkPoints> points; points->SetDataTypeToDouble();
        vtkNew<vtkCellArray> faces;
        for (int k = 0; k <= strips; ++k) {
            const double u = -0.5+double(options_.channels)*k/strips;
            for (double v : {-0.5, options_.rows-0.5}) {
                const auto p = display_pose_.position(scene_.detectorPoint(frame_, u, v)); points->InsertNextPoint(p.data());
            }
        }
        for (int k = 0; k < strips; ++k) {
            const vtkIdType ids[]{2*k, 2*k+2, 2*k+3, 2*k+1}; faces->InsertNextCell(4, ids);
        }
        detector_->SetPoints(points); detector_->SetPolys(faces); detector_->Modified();
        std::vector<Point> rays{s};
        std::vector<std::vector<vtkIdType>> edges;
        for (const auto& uv : std::array<std::array<double, 2>, 5>{{
                {-0.5,-0.5}, {options_.channels-0.5,-0.5},
                {options_.channels-0.5,options_.rows-0.5}, {-0.5,options_.rows-0.5},
                {(options_.channels-1)*0.5,(options_.rows-1)*0.5}}}) {
            rays.push_back(display_pose_.position(scene_.detectorPoint(frame_, uv[0], uv[1])));
            edges.push_back({0, vtkIdType(rays.size()-1)});
        }
        rays_->ShallowCopy(lines(rays, edges));
        const std::string title = "YKCBCT   /   "+options_.geometry+
            (options_.gantry_fixed ? "   /   fixed gantry" : "   /   fixed object");
        title_->SetInput(title.c_str());
        std::ostringstream status;
        status << "Frame " << frame_+1 << " / " << options_.views << "     Source: "
            << std::fixed << std::setprecision(1) << s[0] << ", " << s[1] << ", " << s[2]
            << " mm     Volume: " << options_.size[0] << " x " << options_.size[1] << " x " << options_.size[2]
            << (options_.raw.empty() ? "  (demo phantom)" : "  (RAW input)");
        status_->SetInput(status.str().c_str()); renderer_->ResetCameraClippingRange();
    }

    void resetCamera() {
        const auto& c = options_.center;
        renderer_->GetActiveCamera()->SetFocalPoint(c.data());
        renderer_->GetActiveCamera()->SetPosition(c[0]+options_.sid*2.2,
            c[1]-options_.sid*3.4, c[2]+options_.sid*1.9);
        renderer_->GetActiveCamera()->SetViewUp(0., 0., 1.);
        renderer_->ResetCamera();
        renderer_->ResetCameraClippingRange();
    }

    void configureLocalAxes(vtkAxesActor* axes, const std::array<const char*, 3>& labels) {
        axes->SetTotalLength(options_.sid*0.22, options_.sid*0.22, options_.sid*0.22);
        axes->SetXAxisLabelText(labels[0]); axes->SetYAxisLabelText(labels[1]);
        axes->SetZAxisLabelText(labels[2]);
        axes->SetShaftTypeToCylinder();
        const std::array<Point, 3> colors{{{1.,0.24,0.22},{0.3,0.9,0.36},{0.3,0.6,1.}}};
        std::array<vtkCaptionActor2D*, 3> captions{{axes->GetXAxisCaptionActor2D(),
            axes->GetYAxisCaptionActor2D(), axes->GetZAxisCaptionActor2D()}};
        std::array<vtkProperty*, 3> shafts{{axes->GetXAxisShaftProperty(),
            axes->GetYAxisShaftProperty(), axes->GetZAxisShaftProperty()}};
        std::array<vtkProperty*, 3> tips{{axes->GetXAxisTipProperty(),
            axes->GetYAxisTipProperty(), axes->GetZAxisTipProperty()}};
        for (int j = 0; j < 3; ++j) {
            captions[j]->GetTextActor()->SetTextScaleModeToNone();
            captions[j]->GetCaptionTextProperty()->SetFontSize(14);
            captions[j]->GetCaptionTextProperty()->SetColor(colors[j].data());
            shafts[j]->SetColor(colors[j][0], colors[j][1], colors[j][2]);
            tips[j]->SetColor(colors[j][0], colors[j][1], colors[j][2]);
        }
        renderer_->AddActor(axes);
    }

    void updateLocalAxes(vtkAxesActor* axes, vtkPolyData* data, const Viewer::LocalFrame& frame,
        const std::array<const char*, 3>& labels)
    {
        vtkNew<vtkMatrix4x4> matrix;
        matrix->Identity();
        for (int j = 0; j < 3; ++j) {
            matrix->SetElement(j, 3, frame.origin[j]);
            for (int k = 0; k < 3; ++k) matrix->SetElement(j, k, frame.basis[k][j]);
        }
        vtkNew<vtkTransform> transform; transform->SetMatrix(matrix);
        axes->SetUserTransform(transform);
        std::vector<Point> points{frame.origin};
        for (auto axis : frame.basis) {
            for (int j = 0; j < 3; ++j) axis[j] = frame.origin[j]+axis[j]*options_.sid*0.22;
            points.push_back(axis);
        }
        data->ShallowCopy(lines(points, {{0,1},{0,2},{0,3}}));
        vtkNew<vtkUnsignedCharArray> rgb;
        rgb->SetName("AxisRGB"); rgb->SetNumberOfComponents(3);
        const unsigned char red[]{255,61,56}, green[]{77,230,92}, blue[]{77,153,255};
        rgb->InsertNextTypedTuple(red); rgb->InsertNextTypedTuple(green); rgb->InsertNextTypedTuple(blue);
        data->GetCellData()->SetScalars(rgb);
        vtkNew<vtkStringArray> names; names->SetName("AxisLabels");
        for (const auto* label : labels) names->InsertNextValue(label);
        data->GetFieldData()->AddArray(names);
    }

    void key(const std::string& key) {
        if (key == "Right") setFrame(frame_+1);
        else if (key == "Left") setFrame(frame_-1);
        else if (key == "v" || key == "V") volume_->SetVisibility(!volume_->GetVisibility());
        else if (key == "t" || key == "T") trajectory_actor_->SetVisibility(!trajectory_actor_->GetVisibility());
        else if (key == "l" || key == "L") {
            const bool visible = !source_axes_->GetVisibility();
            source_axes_->SetVisibility(visible); detector_axes_->SetVisibility(visible);
        }
        else if (key == "m" || key == "M") {
            if (!scene_.supportsRelative())
                throw std::runtime_error("SourceOrbit source/detector motion is non-rigid; relative mode is unavailable");
            options_.gantry_fixed = !options_.gantry_fixed;
            rebuildTrajectory(); setFrame(frame_); resetCamera();
        }
        else if (key == "r" || key == "R") resetCamera();
        else if (key == "e" || key == "E") exportScene(options_.export_dir.empty() ? "out/viewer-export" : options_.export_dir);
        else if (key.size() == 1 && key[0] >= '1' && key[0] <= '6') {
            const std::array<std::string, 6> names{"flat-circular","cyl-circular","flat-helical","cyl-helical","source-orbit","nikon-tilted"};
            auto next = options_; next.geometry = names[key[0]-'1'];
            if (next.geometry == "source-orbit") next.gantry_fixed = false;
            Viewer::Scene prepared(next); scene_ = std::move(prepared); options_ = next;
            rebuildTrajectory(); setFrame(frame_); resetCamera();
        }
    }

    void checkInteraction() {
        const auto original_geometry = options_.geometry;
        const bool original_relative = options_.gantry_fixed;
        const int original_frame = frame_;
        const std::array<std::string, 6> names{"flat-circular","cyl-circular","flat-helical","cyl-helical","source-orbit","nikon-tilted"};
        for (int i = 0; i < 6; ++i) {
            const std::string symbol = std::to_string(i+1);
            interactor_->SetKeyEventInformation(0, 0, symbol[0], 0, symbol.c_str());
            interactor_->InvokeEvent(vtkCommand::KeyPressEvent);
            if (options_.geometry != names[i] ||
                detector_->GetNumberOfPoints() != (scene_.cyl.empty() ? 4 : 130))
                throw std::runtime_error("Geometry switch callback failed");
        }
        slider_rep_->SetValue(options_.views-1);
        slider_->InvokeEvent(vtkCommand::InteractionEvent);
        if (frame_ != options_.views-1) throw std::runtime_error("Frame slider callback failed");
        const bool was_visible = volume_->GetVisibility();
        interactor_->SetKeyEventInformation(0, 0, 'v', 0, "v");
        interactor_->InvokeEvent(vtkCommand::KeyPressEvent);
        if (bool(volume_->GetVisibility()) == was_visible) throw std::runtime_error("Volume toggle failed");
        interactor_->InvokeEvent(vtkCommand::KeyPressEvent);
        const bool axes_visible = source_axes_->GetVisibility();
        interactor_->SetKeyEventInformation(0, 0, 'l', 0, "l");
        interactor_->InvokeEvent(vtkCommand::KeyPressEvent);
        if (bool(source_axes_->GetVisibility()) == axes_visible ||
            source_axes_->GetVisibility() != detector_axes_->GetVisibility())
            throw std::runtime_error("Local axes toggle failed");
        interactor_->InvokeEvent(vtkCommand::KeyPressEvent);
        interactor_->SetKeyEventInformation(0, 0, 'm', 0, "m");
        interactor_->InvokeEvent(vtkCommand::KeyPressEvent);
        if (!options_.gantry_fixed) throw std::runtime_error("Relative mode toggle failed");
        setFrame(1);
        const auto fixed_source = scene_.source(0);
        const auto* actual_source = source_->GetCenter();
        for (int j = 0; j < 3; ++j)
            if (std::abs(actual_source[j]-fixed_source[j]) > 1e-4)
                throw std::runtime_error("Source moves in fixed-gantry mode");
        interactor_->InvokeEvent(vtkCommand::KeyPressEvent);
        if (options_.gantry_fixed) throw std::runtime_error("Absolute mode restore failed");
        const auto initial = std::find(names.begin(), names.end(), original_geometry)-names.begin();
        key(std::to_string(initial+1));
        if (options_.gantry_fixed != original_relative) key("m");
        setFrame(original_frame); resetCamera(); window_->Render();
        std::cout << "Interaction checks: model/frame/volume/local axes/relative mode PASS\n";
    }

    void exportScene(const fs::path& folder) {
        fs::create_directories(folder);
        vtkNew<vtkMultiBlockDataSet> geometry;
        geometry->SetNumberOfBlocks(6);
        vtkNew<vtkTransformPolyDataFilter> display_box;
        display_box->SetTransform(display_transform_); display_box->SetInputData(box_->GetOutput()); display_box->Update();
        geometry->SetBlock(0, display_box->GetOutput()); geometry->SetBlock(1, source_->GetOutput());
        geometry->SetBlock(2, detector_); geometry->SetBlock(3, trajectory_);
        geometry->SetBlock(4, source_axes_data_); geometry->SetBlock(5, detector_axes_data_);
        const std::array<const char*, 6> names{"Volume bounds", "Source", "Detector",
            options_.gantry_fixed ? "Object centre trajectory" : "Source trajectory",
            "Source local axes (U,R,V)", "Detector local axes (U,V,N)"};
        for (unsigned int i = 0; i < names.size(); ++i)
            geometry->GetMetaData(i)->Set(vtkCompositeDataSet::NAME(), names[i]);
        vtkNew<vtkXMLMultiBlockDataWriter> writer;
        writer->SetFileName((folder/"geometry.vtm").string().c_str()); writer->SetInputData(geometry);
        if (!writer->Write()) throw std::runtime_error("Geometry export failed");
        vtkNew<vtkXMLImageDataWriter> image_writer;
        vtkNew<vtkImageData> display_image;
        display_image->ShallowCopy(image_);
        vtkNew<vtkMatrix3x3> direction;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) direction->SetElement(row, col, display_pose_.rows[row][col]);
        display_image->SetDirectionMatrix(direction);
        const double* raw_origin = image_->GetOrigin();
        const auto origin = display_pose_.position({raw_origin[0],raw_origin[1],raw_origin[2]});
        display_image->SetOrigin(origin.data());
        image_writer->SetFileName((folder/"volume.vti").string().c_str()); image_writer->SetInputData(display_image);
        if (!image_writer->Write()) throw std::runtime_error("Volume export failed");
        std::ofstream metadata(folder/"scene.txt");
        metadata << "geometry=" << options_.geometry << "\nframe=" << frame_
            << "\nreference=" << (options_.gantry_fixed ? "gantry-fixed" : "world")
            << "\nunits=mm\nvolume_layout=[z][y][x]\ngeometry_builders=YKCBCT\n";
        if (!metadata) throw std::runtime_error("Metadata export failed");
        std::cout << "Export: " << fs::absolute(folder) << '\n';
    }

    Viewer::Options options_;
    Viewer::Scene scene_;
    int frame_ = 0;
    Viewer::DisplayTransform display_pose_{};
    vtkNew<vtkTransform> display_transform_;
    vtkNew<vtkRenderer> renderer_;
    vtkNew<vtkRenderWindow> window_;
    vtkNew<vtkRenderWindowInteractor> interactor_;
    vtkNew<vtkCubeSource> box_;
    vtkNew<vtkSphereSource> source_;
    vtkNew<vtkPolyData> detector_, trajectory_, rays_, source_axes_data_, detector_axes_data_;
    vtkNew<vtkActor> source_actor_;
    vtkSmartPointer<vtkActor> box_actor_, detector_actor_, trajectory_actor_, rays_actor_;
    vtkSmartPointer<vtkImageData> image_;
    vtkNew<vtkVolume> volume_;
    vtkNew<vtkAxesActor> axes_, source_axes_, detector_axes_;
    vtkNew<vtkTextActor> title_, status_, help_;
    vtkNew<vtkSliderRepresentation2D> slider_rep_;
    vtkNew<vtkSliderWidget> slider_;
    vtkNew<vtkCallbackCommand> slider_callback_, key_callback_;
};

int main(int argc, char** argv) {
    Viewer::Options o;
    CLI::App app{"YKCBCT geometry and volume viewer (optional VTK prototype)"};
    app.add_option("--geometry", o.geometry)->check(CLI::IsMember({"flat-circular","cyl-circular","flat-helical","cyl-helical","source-orbit","nikon-tilted"}));
    app.add_option("--views", o.views); app.add_option("--views-per-turn", o.views_per_turn);
    app.add_option("--frame", o.frame); app.add_option("--channels", o.channels); app.add_option("--rows", o.rows);
    app.add_option("--sid", o.sid); app.add_option("--sdd", o.sdd);
    app.add_option("--du", o.du); app.add_option("--dv", o.dv);
    app.add_option("--radius", o.radius); app.add_option("--pitch", o.pitch); app.add_option("--tilt", o.tilt);
    app.add_option("--size", o.size)->expected(3); app.add_option("--spacing", o.spacing)->expected(3);
    app.add_option("--center", o.center)->expected(3);
    app.add_option("--raw", o.raw)->check(CLI::ExistingFile);
    app.add_option("--screenshot", o.screenshot); app.add_option("--export", o.export_dir);
    app.add_flag("--box-only", o.box_only); app.add_flag("--offscreen", o.offscreen);
    app.add_flag("--self-check", o.self_check);
    app.add_flag("--gantry-fixed", o.gantry_fixed, "Freeze the rigid gantry at frame 0 and move the object instead");
    CLI11_PARSE(app, argc, argv);
    try { View view(o); view.run(); return 0; }
    catch (const std::exception& e) { std::cerr << "Viewer: " << e.what() << '\n'; return 1; }
}
