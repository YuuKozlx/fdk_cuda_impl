#include "ParameterPanel.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vtkButtonWidget.h>
#include <vtkCallbackCommand.h>
#include <vtkCoordinate.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSliderRepresentation2D.h>
#include <vtkSliderWidget.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkTexturedButtonRepresentation2D.h>
namespace yk::visualization {
namespace {
    vtkSmartPointer<vtkTextActor> text(vtkRenderer* renderer, const char* value, double x, double y, int font)
    {
        auto actor = vtkSmartPointer<vtkTextActor>::New();
        actor->SetInput(value);
        actor->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
        actor->SetPosition(x, y);
        actor->GetTextProperty()->SetFontSize(font);
        actor->GetTextProperty()->SetColor(.85, .89, .94);
        renderer->AddViewProp(actor);
        return actor;
    }
    int parameterGroup(const std::string& name)
    {
        if (name.find("Object") == 0)
            return 0;
        if (name.find("Source") == 0)
            return 1;
        if (name.find("Detector") == 0)
            return 2;
        return 3;
    }
}
ParameterPanel::ParameterPanel(vtkRenderer* renderer, vtkRenderWindowInteractor* interactor, std::function<void()> action)
    : renderer_(renderer)
    , interactor_(interactor)
    , recompute_(std::move(action))
{
    heading_ = text(renderer, "Geometry parameters", .07, .94, 19);
    const char* names[] = { "Object", "Source", "Detector", "Geometry" };
    for (int i = 0; i < 4; ++i) {
        auto tab = text(renderer, names[i], .07 + .23 * i, .88, 12);
        tabs_.push_back(tab);
    }
    status_ = text(renderer, "", .06, .018, 12);
    button_label_ = text(renderer, "Apply / Recompute", .18, .10, 15);
    button_label_->VisibilityOff();
    vtkNew<vtkImageData> icon;
    icon->SetDimensions(16, 16, 1);
    icon->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
    auto* pixels = static_cast<unsigned char*>(icon->GetScalarPointer());
    for (int i = 0; i < 16 * 16; ++i) {
        pixels[i * 3] = 45;
        pixels[i * 3 + 1] = 175;
        pixels[i * 3 + 2] = 130;
    }
    vtkNew<vtkTexturedButtonRepresentation2D> rep;
    rep->SetNumberOfStates(1);
    rep->SetButtonTexture(0, icon);
    button_ = vtkSmartPointer<vtkButtonWidget>::New();
    button_->SetInteractor(interactor_);
    button_->SetDefaultRenderer(renderer_);
    button_->SetCurrentRenderer(renderer_);
    button_->SetRepresentation(rep);
    vtkNew<vtkCallbackCommand> resize;
    resize->SetClientData(this);
    resize->SetCallback([](vtkObject*, unsigned long, void* data, void*) {auto& p=*static_cast<ParameterPanel*>(data);auto* size=p.interactor_->GetRenderWindow()->GetSize();auto* v=p.renderer_->GetViewport();double x=(v[0]+.09*(v[2]-v[0]))*size[0],y=(v[1]+.11*(v[3]-v[1]))*size[1];double bounds[]={x-8,x+8,y-8,y+8,0,0};p.button_->GetRepresentation()->PlaceWidget(bounds); });
    interactor_->GetRenderWindow()->AddObserver(vtkCommand::WindowResizeEvent, resize);
    vtkNew<vtkCallbackCommand> click;
    click->SetClientData(this);
    click->SetCallback([](vtkObject*, unsigned long, void* data, void*) {auto& p=*static_cast<ParameterPanel*>(data);if(p.enabled_)p.recompute_(); });
    button_->AddObserver(vtkCommand::StateChangedEvent, click);
    vtkNew<vtkCallbackCommand> select;
    select->SetClientData(this);
    select->SetCallback([](vtkObject*, unsigned long, void* data, void*) {auto& p=*static_cast<ParameterPanel*>(data);if(!p.enabled_)return;auto* size=p.interactor_->GetRenderWindow()->GetSize();auto* v=p.renderer_->GetViewport();auto* pos=p.interactor_->GetEventPosition();double x=(double(pos[0])/std::max(size[0],1)-v[0])/(v[2]-v[0]),y=(double(pos[1])/std::max(size[1],1)-v[1])/(v[3]-v[1]);if(x>=0&&x<1&&y>.865&&y<.915){p.showGroup(std::clamp(int((x-.03)/.23),0,3));p.interactor_->GetRenderWindow()->Render();} });
    interactor_->AddObserver(vtkCommand::LeftButtonPressEvent, select, .9);
}
void ParameterPanel::configure(const std::vector<EditableParameter>& specs)
{
    for (auto& e : entries_) {
        e.widget->SetEnabled(0);
        renderer_->RemoveViewProp(e.label);
        renderer_->RemoveViewProp(e.value);
    }
    entries_.clear();
    int rows[4] = {};
    for (const auto& spec : specs) {
        if (!std::isfinite(spec.value) || !std::isfinite(spec.minimum) || !std::isfinite(spec.maximum) || spec.minimum >= spec.maximum || spec.value < spec.minimum || spec.value > spec.maximum)
            throw std::invalid_argument("invalid parameter range");
        Entry e;
        e.spec = spec;
        e.group = parameterGroup(spec.label);
        double y = .77 - .105 * rows[e.group]++;
        std::string label = spec.label;
        for (const char* prefix : { "Object ", "Source ", "Detector " })
            if (label.find(prefix) == 0)
                label = label.substr(std::string(prefix).size());
        e.label = text(renderer_, label.c_str(), .08, y + .025, 14);
        e.value = text(renderer_, "", .91, y + .025, 14);
        e.value->GetTextProperty()->SetJustificationToRight();
        e.value->GetTextProperty()->SetColor(.4, .8, 1);
        e.rep = vtkSmartPointer<vtkSliderRepresentation2D>::New();
        e.rep->SetMinimumValue(spec.minimum);
        e.rep->SetMaximumValue(spec.maximum);
        e.rep->SetValue(spec.value);
        e.rep->SetTitleText("");
        e.rep->ShowSliderLabelOff();
        e.rep->SetSliderLength(.025);
        e.rep->SetSliderWidth(.018);
        e.rep->SetTubeWidth(.004);
        e.rep->SetEndCapWidth(.005);
        e.rep->GetPoint1Coordinate()->SetCoordinateSystemToNormalizedViewport();
        e.rep->GetPoint1Coordinate()->SetValue(.08, y);
        e.rep->GetPoint2Coordinate()->SetCoordinateSystemToNormalizedViewport();
        e.rep->GetPoint2Coordinate()->SetValue(.91, y);
        e.widget = vtkSmartPointer<vtkSliderWidget>::New();
        e.widget->SetInteractor(interactor_);
        e.widget->SetDefaultRenderer(renderer_);
        e.widget->SetCurrentRenderer(renderer_);
        e.widget->SetRepresentation(e.rep);
        e.widget->SetAnimationModeToJump();
        vtkNew<vtkCallbackCommand> change;
        change->SetClientData(this);
        change->SetCallback([](vtkObject*, unsigned long, void* data, void*) {auto& p=*static_cast<ParameterPanel*>(data);for(auto& e:p.entries_)if(e.spec.integral)e.rep->SetValue(std::round(e.rep->GetValue()));p.updateValues();p.status("Draft values. Apply to recalculate all views. Images still show the previous calculation."); });
        e.widget->AddObserver(vtkCommand::InteractionEvent, change);
        entries_.push_back(std::move(e));
    }
    group_ = entries_.empty() ? 0 : entries_.front().group;
    button_label_->SetVisibility(!entries_.empty());
    updateValues();
    showGroup(group_);
}
void ParameterPanel::updateValues()
{
    for (auto& e : entries_) {
        std::ostringstream value;
        value << std::fixed << std::setprecision(e.spec.integral ? 0 : e.spec.label.find("Pixel") != std::string::npos ? 3
                                                                                                                       : 2)
              << e.rep->GetValue();
        e.value->SetInput(value.str().c_str());
    }
}
void ParameterPanel::showGroup(int group)
{
    group_ = group;
    for (int i = 0; i < 4; ++i)
        tabs_[i]->GetTextProperty()->SetColor(i == group ? .35 : .6, i == group ? .8 : .65, i == group ? 1. : .72);
    for (auto& e : entries_) {
        bool show = e.group == group;
        e.label->SetVisibility(show);
        e.value->SetVisibility(show);
        if (interactor_->GetInitialized())
            e.widget->SetEnabled(enabled_ && show);
    }
}
std::vector<double> ParameterPanel::values() const
{
    std::vector<double> out;
    for (auto& e : entries_)
        out.push_back(e.rep->GetValue());
    return out;
}
bool ParameterPanel::set(size_t i, double value)
{
    if (i >= entries_.size() || !std::isfinite(value) || value < entries_[i].spec.minimum || value > entries_[i].spec.maximum)
        return false;
    entries_[i].rep->SetValue(value);
    entries_[i].widget->InvokeEvent(vtkCommand::InteractionEvent);
    return true;
}
void ParameterPanel::enable(bool enabled)
{
    enabled_ = enabled;
    showGroup(group_);
    if (!interactor_->GetInitialized())
        return;
    button_->SetEnabled(enabled && !entries_.empty());
    button_label_->GetTextProperty()->SetColor(enabled ? .9 : .45, enabled ? .95 : .5, enabled ? 1. : .55);
    interactor_->GetRenderWindow()->InvokeEvent(vtkCommand::WindowResizeEvent);
}
void ParameterPanel::status(const std::string& message)
{
    std::istringstream input(message.substr(0, 200));
    std::string word, out;
    int column = 0;
    while (input >> word) {
        if (column + int(word.size()) + 1 > 38) {
            out += '\n';
            column = 0;
        } else if (column) {
            out += ' ';
            ++column;
        }
        out += word;
        column += int(word.size());
    }
    status_->SetInput(out.c_str());
}
void ParameterPanel::apply()
{
    if (enabled_ && !entries_.empty())
        button_->InvokeEvent(vtkCommand::StateChangedEvent);
}
}
