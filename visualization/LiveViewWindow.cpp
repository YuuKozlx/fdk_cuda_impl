#include "LiveView.hpp"
#include "ParameterPanel.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vtkActor.h>
#include <vtkAxesActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCaptionActor2D.h>
#include <vtkCellArray.h>
#include <vtkColorTransferFunction.h>
#include <vtkCubeSource.h>
#include <vtkCylinderSource.h>
#include <vtkFloatArray.h>
#include <vtkGPUVolumeRayCastMapper.h>
#include <vtkImageActor.h>
#include <vtkImageData.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkPNGWriter.h>
#include <vtkPiecewiseFunction.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSliderRepresentation2D.h>
#include <vtkSliderWidget.h>
#include <vtkSmartPointer.h>

#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkTexture.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>
#include <vtkWindowToImageFilter.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLPolyDataWriter.h>

namespace yk::visualization {
namespace {
    class SceneStyle : public vtkInteractorStyleTrackballCamera {
    public:
        static SceneStyle* New();
        vtkTypeMacro(SceneStyle, vtkInteractorStyleTrackballCamera);
        vtkRenderer* scene = nullptr;
        bool inScene()
        {
            auto* pos = Interactor->GetEventPosition();
            auto* size = Interactor->GetRenderWindow()->GetSize();
            auto* v = scene->GetViewport();
            double x = double(pos[0]) / std::max(size[0], 1), y = double(pos[1]) / std::max(size[1], 1);
            return x >= v[0] && x <= v[2] && y >= v[1] && y <= v[3];
        }
        void OnLeftButtonDown() override
        {
            if (inScene())
                Superclass::OnLeftButtonDown();
        }
        void OnMiddleButtonDown() override
        {
            if (inScene())
                Superclass::OnMiddleButtonDown();
        }
        void OnRightButtonDown() override
        {
            if (inScene())
                Superclass::OnRightButtonDown();
        }
        void OnMouseWheelForward() override
        {
            if (inScene())
                Superclass::OnMouseWheelForward();
        }
        void OnMouseWheelBackward() override
        {
            if (inScene())
                Superclass::OnMouseWheelBackward();
        }
    };
    vtkStandardNewMacro(SceneStyle);

    std::vector<unsigned char> gray(const std::vector<float>& values)
    {
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        for (float v : values)
            if (std::isfinite(v)) {
                lo = std::min(lo, double(v));
                hi = std::max(hi, double(v));
            }
        std::vector<unsigned char> rgb(values.size() * 3);
        for (size_t i = 0; i < values.size(); ++i) {
            double v = std::isfinite(values[i]) ? values[i] : lo;
            auto c = static_cast<unsigned char>(hi > lo ? std::clamp((v - lo) / (hi - lo), 0., 1.) * 255. : 128);
            for (int j = 0; j < 3; ++j)
                rgb[i * 3 + j] = c;
        }
        return rgb;
    }
    Point sub(Point a, Point b)
    {
        for (int i = 0; i < 3; ++i)
            a[i] -= b[i];
        return a;
    }
    Point unit(Point a)
    {
        const double n = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        if (!std::isfinite(n) || n < 1e-12)
            throw std::invalid_argument("degenerate display frame");
        for (auto& x : a)
            x /= n;
        return a;
    }
    Point cross(Point a, Point b) { return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] }; }
    vtkSmartPointer<vtkMatrix4x4> matrix(const Pose& p)
    {
        auto m = vtkSmartPointer<vtkMatrix4x4>::New();
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                m->SetElement(r, c, p[r * 4 + c]);
        return m;
    }
    Pose axesPose(Point origin, const std::array<Point, 3>& axes)
    {
        auto m = identityPose();
        for (int r = 0; r < 3; ++r) {
            m[r * 4 + 3] = origin[r];
            for (int c = 0; c < 3; ++c)
                m[r * 4 + c] = axes[c][r];
        }
        return m;
    }
    vtkSmartPointer<vtkPolyData> trajectory(const std::vector<Point>& path, bool dashed = false)
    {
        vtkNew<vtkPoints> points;
        points->SetDataTypeToDouble();
        vtkNew<vtkCellArray> cells;
        for (auto point : path) {
            for (double x : point)
                if (!std::isfinite(x))
                    throw std::invalid_argument("nonfinite trajectory");
            points->InsertNextPoint(point.data());
        }
        if (dashed) {
            for (vtkIdType i = 0; i + 1 < vtkIdType(path.size()); ++i)
                if (i % 4 < 2) {
                    vtkIdType ids[] = { i, i + 1 };
                    cells->InsertNextCell(2, ids);
                }
        } else if (path.size() > 1) {
            cells->InsertNextCell(vtkIdType(path.size()));
            for (vtkIdType i = 0; i < vtkIdType(path.size()); ++i)
                cells->InsertCellPoint(i);
        }
        auto mesh = vtkSmartPointer<vtkPolyData>::New();
        mesh->SetPoints(points);
        mesh->SetLines(cells);
        return mesh;
    }
}
struct LiveView::Impl {
    struct CachedFrame {
        FrameGeometry geometry;
        int width = 0, height = 0;
        std::vector<unsigned char> rgb;
        std::vector<float> energy, attenuation;
        std::string description;
    };
    std::map<int, CachedFrame> frames;
    vtkNew<vtkSliderRepresentation2D> slider_rep;
    vtkNew<vtkSliderWidget> slider;
    vtkNew<vtkActor> source_trail;
    vtkNew<vtkActor> detector_trail;
    vtkSmartPointer<vtkPolyData> source_path, detector_path;
    bool following = true, browsing = true, recompute = false, interactive = false, editing = false, beam_visible = true;
    LiveViewOptions options;
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderer> slider_renderer;
    vtkNew<vtkRenderer> energy_renderer, attenuation_renderer, parameter_renderer;
    vtkNew<vtkImageActor> energy_actor, attenuation_actor;
    vtkNew<vtkImageData> energy_image, attenuation_image;
    vtkNew<vtkTextActor> energy_title, attenuation_title;
    std::unique_ptr<ParameterPanel> parameters;
    vtkNew<vtkRenderWindow> window;
    vtkNew<vtkRenderWindowInteractor> interactor;
    vtkNew<vtkImageData> labels;
    vtkNew<vtkVolume> volume;
    vtkNew<vtkGPUVolumeRayCastMapper> volume_mapper;
    vtkNew<vtkActor> box_actor;
    vtkNew<vtkActor> detector_actor;
    vtkNew<vtkActor> source_actor;
    vtkNew<vtkActor> beam_actor;
    vtkNew<vtkPolyData> beam;
    vtkSmartPointer<vtkPolyData> source_body;
    vtkNew<vtkPolyData> detector;
    vtkNew<vtkTexture> texture;
    vtkNew<vtkImageData> texture_image;
    vtkNew<vtkTextActor> text;
    vtkNew<vtkAxesActor> object_axes;
    vtkNew<vtkAxesActor> detector_axes;
    vtkNew<vtkAxesActor> source_axes;
    bool closed = false, initialized = false, has_volume = false, has_frame = false;
    std::chrono::steady_clock::time_point last_status_render {};
    FrameGeometry last_frame;
    std::chrono::steady_clock::time_point last_render {};
    double axis_length = 10;
    explicit Impl(const LiveViewOptions& o)
        : options(o)
    {
        if (o.texture_max_size < 1 || o.texture_max_size > 8192 || !std::isfinite(o.max_fps) || o.max_fps <= 0 || o.max_fps > 120)
            throw std::invalid_argument("invalid live display options");
        renderer->SetBackground(.04, .06, .1);
        window->AddRenderer(renderer);
        window->SetSize(1600, 1000);
        window->SetMultiSamples(0);
        window->SetNumberOfLayers(2);
        slider_renderer->SetLayer(1);
        slider_renderer->SetInteractive(false);
        window->AddRenderer(slider_renderer);
        renderer->SetViewport(.18, .10, .72, 1);
        parameter_renderer->SetViewport(0, 0, .18, 1);
        energy_renderer->SetViewport(.72, .5, 1, 1);
        attenuation_renderer->SetViewport(.72, 0, 1, .5);
        parameter_renderer->SetBackground(.075, .095, .12);
        energy_renderer->SetBackground(.025, .035, .05);
        attenuation_renderer->SetBackground(.04, .05, .065);
        window->AddRenderer(parameter_renderer);
        window->AddRenderer(energy_renderer);
        window->AddRenderer(attenuation_renderer);
        window->SetAlphaBitPlanes(1);
        renderer->UseDepthPeelingOn();
        renderer->UseDepthPeelingForVolumesOn();
        energy_renderer->AddActor(energy_actor);
        attenuation_renderer->AddActor(attenuation_actor);
        energy_actor->VisibilityOff();
        attenuation_actor->VisibilityOff();
        energy_title->SetInput("Energy (not computed)");
        attenuation_title->SetInput("Attenuation (not computed)");
        for (auto* title : { energy_title.GetPointer(), attenuation_title.GetPointer() }) {
            title->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
            title->SetPosition(.03, .88);
            title->GetTextProperty()->SetFontSize(18);
        }
        energy_renderer->AddViewProp(energy_title);
        attenuation_renderer->AddViewProp(attenuation_title);
        energy_renderer->GetActiveCamera()->ParallelProjectionOn();
        attenuation_renderer->GetActiveCamera()->ParallelProjectionOn();
        window->SetWindowName("YKCBCT - live spectral projection");
        window->SetOffScreenRendering(o.offscreen);
        interactor->SetRenderWindow(window);
        vtkNew<SceneStyle> camera;
        camera->scene = renderer;
        interactor->SetInteractorStyle(camera);
        parameters = std::make_unique<ParameterPanel>(parameter_renderer, interactor, [this] {recompute=true;if(interactive)interactor->TerminateApp(); });
        vtkNew<vtkCallbackCommand> exit;
        exit->SetClientData(this);
        exit->SetCallback([](vtkObject*, unsigned long, void* data, void*) { static_cast<Impl*>(data)->closed = true; });
        interactor->AddObserver(vtkCommand::ExitEvent, exit);
        vtkNew<vtkPolyDataMapper> dm;
        dm->SetInputData(detector);
        detector_actor->SetMapper(dm);
        texture->SetInputData(texture_image);
        texture->InterpolateOff();
        texture->RepeatOff();
        texture->EdgeClampOn();
        detector_actor->SetTexture(texture);
        detector_actor->GetProperty()->LightingOff();
        renderer->AddActor(detector_actor);
        renderer->AddActor(source_actor);
        renderer->AddActor(box_actor);
        renderer->AddVolume(volume);
        vtkNew<vtkPolyDataMapper> bm;
        bm->SetInputData(beam);
        beam_actor->SetMapper(bm);
        beam_actor->GetProperty()->SetColor(1, .8, .35);
        beam_actor->GetProperty()->SetOpacity(.08);
        beam_actor->GetProperty()->LightingOff();
        renderer->AddActor(beam_actor);
        renderer->AddActor(object_axes);
        renderer->AddActor(detector_axes);
        renderer->AddActor(source_axes);
        for (auto* axes : { object_axes.GetPointer(), detector_axes.GetPointer(), source_axes.GetPointer() }) {
            const std::array<vtkCaptionActor2D*, 3> captions = { axes->GetXAxisCaptionActor2D(), axes->GetYAxisCaptionActor2D(), axes->GetZAxisCaptionActor2D() };
            for (int i = 0; i < 3; ++i) {
                captions[i]->GetTextActor()->SetTextScaleModeToNone();
                captions[i]->GetCaptionTextProperty()->SetFontSize(14);
                captions[i]->GetCaptionTextProperty()->SetColor(i == 0 ? 1. : .4, i == 1 ? 1. : .4, i == 2 ? 1. : .4);
            }
        }
        detector_axes->SetXAxisLabelText("D-U");
        detector_axes->SetYAxisLabelText("D-V");
        detector_axes->SetZAxisLabelText("D-N");
        source_axes->SetXAxisLabelText("S-U");
        source_axes->SetYAxisLabelText("S-R");
        source_axes->SetZAxisLabelText("S-V");
        text->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
        text->SetPosition(.015, .095);
        text->GetTextProperty()->SetFontSize(14);
        text->GetTextProperty()->SetColor(.9, .95, 1);
        renderer->AddViewProp(text);
        renderer->AddActor(source_trail);
        renderer->AddActor(detector_trail);
        source_trail->GetProperty()->SetColor(1, .65, .15);
        detector_trail->GetProperty()->SetColor(.2, .85, 1);
        for (auto* actor : { source_trail.GetPointer(), detector_trail.GetPointer() }) {
            actor->GetProperty()->SetLineWidth(2);
            actor->GetProperty()->LightingOff();
        }
        slider_rep->SetMinimumValue(0);
        slider_rep->SetMaximumValue(1);
        slider_rep->SetValue(0);
        slider_rep->SetTitleText("Computed scan frame");
        slider_rep->ShowSliderLabelOff();
        slider_rep->SetTitleHeight(.014);
        slider_rep->SetSliderLength(.018);
        slider_rep->SetSliderWidth(.014);
        slider_rep->SetEndCapLength(.004);
        slider_rep->SetEndCapWidth(.012);
        slider_rep->GetSliderProperty()->SetColor(.35, .8, 1);
        slider_rep->GetSelectedProperty()->SetColor(.5, .9, 1);
        slider_rep->GetTubeProperty()->SetColor(.22, .3, .4);
        slider_rep->GetCapProperty()->SetColor(.3, .4, .5);
        slider_rep->GetTitleProperty()->SetBold(false);

        slider_rep->SetTubeWidth(.006);
        slider_rep->GetPoint1Coordinate()->SetCoordinateSystemToNormalizedDisplay();
        slider_rep->GetPoint1Coordinate()->SetValue(.234, .075);
        slider_rep->GetPoint2Coordinate()->SetCoordinateSystemToNormalizedDisplay();
        slider_rep->GetPoint2Coordinate()->SetValue(.666, .075);
        slider->SetInteractor(interactor);
        slider->SetCurrentRenderer(slider_renderer);
        slider->SetDefaultRenderer(slider_renderer);
        slider->SetRepresentation(slider_rep);
        slider->SetAnimationModeToJump();
        vtkNew<vtkCallbackCommand> navigation;
        navigation->SetClientData(this);
        navigation->SetCallback([](vtkObject*, unsigned long, void* data, void*) {auto& p=*static_cast<Impl*>(data);
            p.navigate(int(std::lround(p.slider_rep->GetValue()))); });
        slider->AddObserver(vtkCommand::InteractionEvent, navigation);
        vtkNew<vtkCallbackCommand> keys;
        keys->SetClientData(this);
        keys->SetCallback([](vtkObject*, unsigned long, void* data, void*) {
            auto& p = *static_cast<Impl*>(data);
            const std::string key = p.interactor->GetKeySym();
            if (key == "Return" || key == "KP_Enter") {
                p.parameters->apply();
                return;
            }
            if (!p.browsing || p.frames.empty())
                return;
            if (key == "End") {
                p.following = true;
                p.renderFrame(p.frames.rbegin()->first);
            } else if (key == "Left" || key == "Right") {
                auto it = p.frames.find(p.last_frame.view);
                if (it == p.frames.end())
                    return;
                if (key == "Left" && it != p.frames.begin())
                    --it;
                else if (key == "Right" && std::next(it) != p.frames.end())
                    ++it;
                p.navigate(it->first);
            } else if (key == "t" || key == "T") {
                const bool show = !p.source_trail->GetVisibility();
                p.source_trail->SetVisibility(show);
                p.detector_trail->SetVisibility(show);
                p.window->Render();
            }
        });
        interactor->AddObserver(vtkCommand::KeyPressEvent, keys);
        vtkNew<vtkCallbackCommand> beam_key;
        beam_key->SetClientData(this);
        beam_key->SetCallback([](vtkObject*, unsigned long, void* data, void*) {auto& p=*static_cast<Impl*>(data);std::string k=p.interactor->GetKeySym();if(k=="b"||k=="B"){p.beam_visible=!p.beam_visible;p.beam_actor->SetVisibility(p.beam_visible);p.window->Render();} });
        interactor->AddObserver(vtkCommand::KeyPressEvent, beam_key);
    }
    bool navigate(int view)
    {
        if (closed || !browsing || frames.find(view) == frames.end())
            return false;
        following = false;
        if (has_frame && last_frame.view == view)
            return true;
        renderFrame(view);
        return true;
    }
    void renderFrame(int view);
    void events()
    {
        if (initialized && !options.offscreen && !closed)
            interactor->ProcessEvents();
    }
    void panel(vtkRenderer* r, vtkImageActor* actor, vtkImageData* image, vtkTextActor* title, const std::vector<float>& values, int w, int h, const char* name, int view)
    {
        if (values.empty()) {
            actor->VisibilityOff();
            title->SetInput((std::string(name) + " (not provided)").c_str());
            return;
        }
        auto rgb = gray(values);
        image->SetDimensions(w, h, 1);
        image->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
        std::memcpy(image->GetScalarPointer(), rgb.data(), rgb.size());
        image->Modified();
        actor->SetInputData(image);
        actor->InterpolateOff();
        actor->VisibilityOn();
        auto* c = r->GetActiveCamera();
        c->SetPosition((w - 1) * .5, (h - 1) * .5, 1000);
        c->SetFocalPoint((w - 1) * .5, (h - 1) * .5, 0);
        c->SetViewUp(0, 1, 0);
        r->ResetCamera();
        auto range = std::minmax_element(values.begin(), values.end());
        std::ostringstream label;
        label << name << " | frame " << view << "\nrange " << *range.first << " .. " << *range.second;
        title->SetInput(label.str().c_str());
    }
    void save()
    {
        if (options.snapshot_directory.empty() || !has_frame)
            return;
        namespace fs = std::filesystem;
        fs::create_directories(options.snapshot_directory);
        const auto path = [&](const char* name) { return (fs::path(options.snapshot_directory) / name).string(); };
        const auto* window_size = window->GetSize();
        if (window_size[0] < 1 || window_size[1] < 1)
            return;
        vtkNew<vtkWindowToImageFilter> capture;
        capture->SetInput(window);
        capture->ReadFrontBufferOff();
        capture->Update();
        vtkNew<vtkPNGWriter> png;
        png->SetFileName(path("live.png").c_str());
        png->SetInputConnection(capture->GetOutputPort());
        const auto* snapshot_size = capture->GetOutput()->GetDimensions();
        if (snapshot_size[0] > 0 && snapshot_size[1] > 0)
            png->Write();
        vtkNew<vtkXMLImageDataWriter> image;
        image->SetFileName(path("labels.vti").c_str());
        image->SetInputData(labels);
        image->Write();
        image->SetFileName(path("texture.vti").c_str());
        image->SetInputData(texture_image);
        image->Write();
        vtkNew<vtkXMLPolyDataWriter> mesh;
        mesh->SetFileName(path("detector.vtp").c_str());
        mesh->SetInputData(detector);
        mesh->Write();
        mesh->SetFileName(path("beam.vtp").c_str());
        mesh->SetInputData(beam);
        mesh->Write();
        if (source_body) {
            vtkNew<vtkTransform> t;
            t->SetMatrix(source_actor->GetUserMatrix());
            vtkNew<vtkTransformPolyDataFilter> world;
            world->SetInputData(source_body);
            world->SetTransform(t);
            world->Update();
            mesh->SetFileName(path("source-body.vtp").c_str());
            mesh->SetInputData(world->GetOutput());
            mesh->Write();
        }
        const auto& saved = frames.at(last_frame.view);
        for (int i = 0; i < 2; ++i) {
            const auto& values = i ? saved.attenuation : saved.energy;
            if (values.empty())
                continue;
            vtkNew<vtkImageData> raw;
            raw->SetDimensions(saved.width, saved.height, 1);
            raw->AllocateScalars(VTK_FLOAT, 1);
            std::memcpy(raw->GetScalarPointer(), values.data(), values.size() * sizeof(float));
            image->SetFileName(path(i ? "attenuation.vti" : "energy.vti").c_str());
            image->SetInputData(raw);
            image->Write();
        }
        if (source_path) {
            mesh->SetFileName(path("source-trajectory.vtp").c_str());
            mesh->SetInputData(source_path);
            mesh->Write();
        }
        if (detector_path) {
            mesh->SetFileName(path("detector-trajectory.vtp").c_str());
            mesh->SetInputData(detector_path);
            mesh->Write();
        }
        std::ofstream meta(path("scene.txt"));
        meta.precision(17);
        meta << "view " << last_frame.view << "\nsource ";
        for (double x : last_frame.source)
            meta << x << ' ';
        meta << "\nobject_pose ";
        for (double x : last_frame.object_pose)
            meta << x << ' ';
        meta << "\ndetector_center ";
        for (double x : last_frame.detector_center)
            meta << x << ' ';
        meta << "\ndetector_axes ";
        for (auto a : last_frame.detector_axes)
            for (double x : a)
                meta << x << ' ';
        meta << "\n";
    }
};
LiveView::LiveView(const LiveViewOptions& o)
    : impl_(std::make_unique<Impl>(o))
{
}
LiveView::~LiveView() = default;
void LiveView::setVolume(const LabelVolume& input)
{
    auto& p = *impl_;
    size_t n = 1;
    for (int i = 0; i < 3; ++i) {
        if (input.dimensions[i] <= 0 || !std::isfinite(input.spacing[i]) || input.spacing[i] <= 0)
            throw std::invalid_argument("invalid display volume grid");
        if (n > std::numeric_limits<size_t>::max() / size_t(input.dimensions[i]))
            throw std::invalid_argument("display volume overflow");
        n *= input.dimensions[i];
    }
    if (input.labels.size() != n)
        throw std::invalid_argument("display label volume size mismatch");
    p.labels->SetDimensions(input.dimensions.data());
    p.labels->SetSpacing(input.spacing.data());
    Point origin {};
    for (int i = 0; i < 3; ++i)
        origin[i] = -.5 * (input.dimensions[i] - 1) * input.spacing[i];
    p.labels->SetOrigin(origin.data());
    p.labels->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
    std::memcpy(p.labels->GetScalarPointer(), input.labels.data(), n);
    p.labels->Modified();
    vtkNew<vtkColorTransferFunction> colors;
    vtkNew<vtkPiecewiseFunction> opacity;
    std::array<std::array<double, 4>, 256> palette {};
    for (const auto& m : input.materials) {
        for (double x : m.rgba)
            if (!std::isfinite(x) || x < 0 || x > 1)
                throw std::invalid_argument("invalid material display color");
        palette[m.label] = m.rgba;
    }
    for (int i = 0; i < 256; ++i) {
        const auto& c = palette[i];
        colors->AddRGBPoint(i, c[0], c[1], c[2]);
        opacity->AddPoint(i, c[3]);
    }
    vtkNew<vtkVolumeProperty> property;
    property->SetColor(colors);
    property->SetScalarOpacity(opacity);
    property->SetInterpolationTypeToNearest();
    property->ShadeOff();
    property->SetScalarOpacityUnitDistance(*std::min_element(input.spacing.begin(), input.spacing.end()));
    p.volume_mapper->SetInputData(p.labels);
    p.volume_mapper->SetBlendModeToComposite();
    p.volume->SetMapper(p.volume_mapper);
    p.volume->SetProperty(property);
    vtkNew<vtkCubeSource> box;
    box->SetXLength(input.dimensions[0] * input.spacing[0]);
    box->SetYLength(input.dimensions[1] * input.spacing[1]);
    box->SetZLength(input.dimensions[2] * input.spacing[2]);
    vtkNew<vtkPolyDataMapper> bm;
    bm->SetInputConnection(box->GetOutputPort());
    p.box_actor->SetMapper(bm);
    p.box_actor->GetProperty()->SetRepresentationToWireframe();
    p.box_actor->GetProperty()->SetColor(.6, .7, .8);
    p.axis_length = std::max(1., .2 * input.dimensions[0] * input.spacing[0]);
    for (auto* axes : { p.object_axes.GetPointer(), p.detector_axes.GetPointer(), p.source_axes.GetPointer() })
        axes->SetTotalLength(p.axis_length, p.axis_length, p.axis_length);
    p.has_volume = true;
    invalidateResults();
}
void LiveView::invalidateResults()
{
    auto& p = *impl_;
    p.frames.clear();
    p.following = true;
    p.has_frame = false;
    p.slider->SetEnabled(0);
    p.energy_actor->VisibilityOff();
    p.attenuation_actor->VisibilityOff();
    p.detector_actor->VisibilityOff();
    p.source_actor->VisibilityOff();
    p.beam_actor->VisibilityOff();
    p.energy_title->SetInput("Energy: computing");
    p.attenuation_title->SetInput("Attenuation: computing");
    if (p.initialized)
        p.window->Render();
}
void LiveView::setParameters(const std::vector<EditableParameter>& values) { impl_->parameters->configure(values); }
std::vector<double> LiveView::parameterValues() const { return impl_->parameters->values(); }
bool LiveView::setParameter(size_t i, double value) { return impl_->parameters->set(i, value); }
void LiveView::setEditingEnabled(bool enabled)
{
    auto& p = *impl_;
    p.editing = enabled;
    p.browsing = enabled;
    p.parameters->enable(enabled);
    if (p.initialized) {
        p.slider->SetEnabled(enabled && p.frames.size() > 1);
        if (!p.closed)
            p.window->Render();
    }
}
void LiveView::setStatus(const std::string& text)
{
    auto& p = *impl_;
    p.parameters->status(text);
    if (p.initialized && !p.closed) {
        const auto now = std::chrono::steady_clock::now();
        if (p.browsing || std::chrono::duration<double>(now - p.last_status_render).count() >= .25) {
            p.window->Render();
            p.last_status_render = now;
        }
        p.events();
    }
}
void LiveView::requestRecompute() { impl_->parameters->apply(); }
bool LiveView::takeRecomputeRequest()
{
    bool v = impl_->recompute;
    impl_->recompute = false;
    return v;
}
void LiveView::setTrajectory(const std::vector<Point>& sources, const std::vector<Point>& centers)
{
    if (sources.size() != centers.size())
        throw std::invalid_argument("trajectory count mismatch");
    auto& p = *impl_;
    p.source_path = trajectory(sources);
    p.detector_path = trajectory(centers, true);
    vtkNew<vtkPolyDataMapper> sm;
    sm->SetInputData(p.source_path);
    p.source_trail->SetMapper(sm);
    vtkNew<vtkPolyDataMapper> dm;
    dm->SetInputData(p.detector_path);
    p.detector_trail->SetMapper(dm);
}
bool LiveView::isOpen() const { return !impl_->closed; }
bool LiveView::selectFrame(int view)
{
    auto& p = *impl_;
    if (p.closed || !p.browsing || p.frames.find(view) == p.frames.end())
        return false;
    p.slider_rep->SetValue(view);
    p.slider->InvokeEvent(vtkCommand::InteractionEvent);
    return displayedFrame() == view;
}
int LiveView::displayedFrame() const { return impl_->has_frame ? impl_->last_frame.view : -1; }
std::size_t LiveView::cachedFrameCount() const { return impl_->frames.size(); }
bool LiveView::wantsFrame()
{
    auto& p = *impl_;
    p.events();
    return !p.closed && (!p.has_frame || std::chrono::duration<double>(std::chrono::steady_clock::now() - p.last_render).count() >= 1. / p.options.max_fps);
}
void LiveView::update(const FrameGeometry& frame, int width, int height, const std::vector<float>& pixels, const std::string& description, bool force_render)
{
    updateProjections(frame, width, height, pixels, {}, {}, description, force_render);
}
void LiveView::updateProjections(const FrameGeometry& frame, int width, int height, const std::vector<float>& pixels, const std::vector<float>& energy, const std::vector<float>& attenuation, const std::string& description, bool force_render)
{
    auto& p = *impl_;
    p.events();
    if (p.closed)
        return;
    if ((!energy.empty() && energy.size() != pixels.size()) || (!attenuation.empty() && attenuation.size() != pixels.size()))
        throw std::invalid_argument("panel dimensions mismatch");
    if (!p.has_volume || width <= 0 || height <= 0 || pixels.size() != size_t(width) * height || frame.detector.size() < 4 || frame.detector.size() % 2)
        throw std::invalid_argument("invalid live projection frame");
    for (double x : frame.object_pose)
        if (!std::isfinite(x))
            throw std::invalid_argument("nonfinite object pose");
    for (auto point : frame.detector)
        for (double x : point)
            if (!std::isfinite(x))
                throw std::invalid_argument("nonfinite detector geometry");
    for (double x : frame.source)
        if (!std::isfinite(x))
            throw std::invalid_argument("nonfinite source geometry");
    const double scale = std::min(1., double(p.options.texture_max_size) / std::max(width, height));
    const int tw = std::max(1, int(std::ceil(width * scale))), th = std::max(1, int(std::ceil(height * scale)));
    Impl::CachedFrame saved;
    saved.geometry = frame;
    saved.width = tw;
    saved.height = th;
    saved.description = description;
    saved.rgb.resize(size_t(tw) * th * 3);
    if (!energy.empty())
        saved.energy.resize(size_t(tw) * th);
    if (!attenuation.empty())
        saved.attenuation.resize(size_t(tw) * th);
    const bool energy_is_texture = !energy.empty() && energy.data() == pixels.data();
    const bool attenuation_is_texture = !attenuation.empty() && attenuation.data() == pixels.data();
    std::vector<double> means(size_t(tw) * th);
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    // One pass reads all source pixels and downsamples the three display channels.
    for (int y = 0; y < th; ++y)
        for (int x = 0; x < tw; ++x) {
            double sum = 0, energy_sum = 0, attenuation_sum = 0;
            size_t count = 0, energy_count = 0, attenuation_count = 0;
            for (int sy = y * height / th; sy < (y + 1) * height / th; ++sy)
                for (int sx = x * width / tw; sx < (x + 1) * width / tw; ++sx) {
                    const size_t i = size_t(sy) * width + sx;
                    const float v = pixels[i];
                    if (std::isfinite(v)) {
                        sum += v;
                        ++count;
                        lo = std::min(lo, double(v));
                        hi = std::max(hi, double(v));
                    }
                    if (!energy.empty() && !energy_is_texture) {
                        const float e = energy[i];
                        if (std::isfinite(e)) {
                            energy_sum += e;
                            ++energy_count;
                        }
                    }
                    if (!attenuation.empty() && !attenuation_is_texture) {
                        const float a = attenuation[i];
                        if (std::isfinite(a)) {
                            attenuation_sum += a;
                            ++attenuation_count;
                        }
                    }
                }
            const size_t i = size_t(y) * tw + x;
            means[i] = count ? sum / count : std::numeric_limits<double>::quiet_NaN();
            if (!energy.empty())
                saved.energy[i] = energy_is_texture ? float(count ? sum / count : 0) : float(energy_count ? energy_sum / energy_count : 0);
            if (!attenuation.empty())
                saved.attenuation[i] = attenuation_is_texture ? float(count ? sum / count : 0) : float(attenuation_count ? attenuation_sum / attenuation_count : 0);
        }
    if (!std::isfinite(lo)) {
        lo = 0;
        hi = 0;
    }
    for (size_t i = 0; i < means.size(); ++i) {
        const double v = std::isfinite(means[i]) ? means[i] : lo;
        const auto gray = static_cast<unsigned char>(hi > lo ? std::clamp((v - lo) / (hi - lo), 0., 1.) * 255. : 128);
        for (int c = 0; c < 3; ++c)
            saved.rgb[i * 3 + c] = gray;
    }
    p.frames[frame.view] = std::move(saved);
    p.slider_rep->SetMinimumValue(p.frames.begin()->first);
    p.slider_rep->SetMaximumValue(std::max(p.frames.begin()->first + 1, p.frames.rbegin()->first));
    if (p.following && (force_render || !p.has_frame || (p.browsing && wantsFrame())))
        p.renderFrame(frame.view);
}
void LiveView::Impl::renderFrame(int view)
{
    auto& p = *this;
    if (p.closed)
        return;
    const auto& saved = p.frames.at(view);
    const auto& frame = saved.geometry;
    const int tw = saved.width, th = saved.height;
    vtkNew<vtkPoints> points;
    points->SetDataTypeToDouble();
    vtkNew<vtkCellArray> quads;
    vtkNew<vtkFloatArray> uv;
    uv->SetNumberOfComponents(2);
    const int columns = int(frame.detector.size() / 2);
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < columns; ++col) {
            points->InsertNextPoint(frame.detector[row * columns + col].data());
            uv->InsertNextTuple2(double(col) / (columns - 1), row);
        }
    for (int col = 0; col < columns - 1; ++col) {
        vtkIdType ids[] = { col, col + 1, columns + col + 1, columns + col };
        quads->InsertNextCell(4, ids);
    }
    p.detector->SetPoints(points);
    p.detector->SetPolys(quads);
    p.detector->GetPointData()->SetTCoords(uv);
    p.detector->Modified();
    p.texture_image->SetDimensions(tw, th, 1);
    p.texture_image->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
    std::memcpy(p.texture_image->GetScalarPointer(), saved.rgb.data(), saved.rgb.size());
    p.texture_image->Modified();
    p.texture->Modified();

    const auto pose = matrix(frame.object_pose);
    p.volume->SetUserMatrix(pose);
    p.box_actor->SetUserMatrix(pose);
    p.object_axes->SetUserMatrix(pose);
    p.detector_axes->SetUserMatrix(matrix(axesPose(frame.detector_center, frame.detector_axes)));
    const auto ray = unit(sub(frame.detector_center, frame.source));
    auto u = frame.detector_axes[0];
    const double dot = u[0] * ray[0] + u[1] * ray[1] + u[2] * ray[2];
    for (int i = 0; i < 3; ++i)
        u[i] -= dot * ray[i];
    u = unit(u);
    auto source_pose = matrix(axesPose(frame.source, { u, ray, unit(cross(u, ray)) }));
    p.source_axes->SetUserMatrix(source_pose);
    double height = std::max(4., p.axis_length * .65);
    vtkNew<vtkCylinderSource> tube;
    tube->SetRadius(std::max(1., p.axis_length * .18));
    tube->SetHeight(height);
    tube->SetCenter(0, -height * .5, 0);
    tube->SetResolution(48);
    tube->CappingOn();
    tube->Update();
    p.source_body = tube->GetOutput();
    vtkNew<vtkPolyDataMapper> sm;
    sm->SetInputData(p.source_body);
    p.source_actor->SetMapper(sm);
    p.source_actor->SetUserMatrix(source_pose);
    p.source_actor->GetProperty()->SetColor(1, .65, .15);
    p.source_actor->GetProperty()->SetAmbient(.35);
    p.source_actor->GetProperty()->SetDiffuse(.65);
    vtkNew<vtkPoints> beam_points;
    beam_points->SetDataTypeToDouble();
    beam_points->InsertNextPoint(frame.source.data());
    for (int col = 0; col < columns; ++col)
        beam_points->InsertNextPoint(frame.detector[col].data());
    for (int col = columns - 1; col >= 0; --col)
        beam_points->InsertNextPoint(frame.detector[columns + col].data());
    vtkNew<vtkCellArray> faces;
    for (vtkIdType i = 0; i < 2 * columns; ++i) {
        vtkIdType ids[] = { 0, i + 1, (i + 1) % (2 * columns) + 1 };
        faces->InsertNextCell(3, ids);
    }
    p.beam->SetPoints(beam_points);
    p.beam->SetPolys(faces);
    p.beam->Modified();
    p.source_actor->VisibilityOn();
    p.detector_actor->VisibilityOn();
    p.beam_actor->SetVisibility(p.beam_visible);
    p.panel(p.energy_renderer, p.energy_actor, p.energy_image, p.energy_title, saved.energy, tw, th, "Energy signal", frame.view);
    p.panel(p.attenuation_renderer, p.attenuation_actor, p.attenuation_image, p.attenuation_title, saved.attenuation, tw, th, "Attenuation -log(I/Iair)", frame.view);
    std::ostringstream caption;
    caption << saved.description << "\nframe " << frame.view << " | computed " << p.frames.size() << " | source " << frame.source[0] << ", " << frame.source[1] << ", " << frame.source[2] << " mm\ntexture " << tw << " x " << th << " | volume full resolution"
            << "\nSlider / Left / Right: scan frame | End: latest | T: trajectories\nOrange: source path | Cyan: detector center path";
    p.text->SetInput(caption.str().c_str());
    if (!p.initialized) {
        auto* camera = p.renderer->GetActiveCamera();
        camera->SetPosition(1, -1, .8);
        camera->SetFocalPoint(0, 0, 0);
        camera->SetViewUp(0, 0, 1);
        p.renderer->ResetCamera();
        p.interactor->Initialize();
        p.initialized = true;
        p.parameters->enable(p.editing);
    }
    p.slider_rep->SetValue(frame.view);
    if (p.browsing && p.frames.size() > 1)
        p.slider->SetEnabled(1);
    p.last_frame = frame;
    p.has_frame = true;
    p.renderer->ResetCameraClippingRange();
    p.window->Render();
    p.last_render = std::chrono::steady_clock::now();
}
void LiveView::finish()
{
    auto& p = *impl_;
    if (p.closed)
        return;
    if (p.following && !p.frames.empty() && (!p.has_frame || p.last_frame.view != p.frames.rbegin()->first))
        p.renderFrame(p.frames.rbegin()->first);
    p.save();
    if (p.options.hold_open && !p.options.offscreen && p.initialized && !p.recompute) {
        p.interactive = true;
        p.interactor->SetDone(false);
        p.interactor->Start();
        p.interactive = false;
    }
}
} // namespace yk::visualization
