#pragma once
#include "LiveView.hpp"
#include <functional>
#include <vtkSmartPointer.h>
class vtkRenderer;
class vtkRenderWindowInteractor;
class vtkSliderWidget;
class vtkSliderRepresentation2D;
class vtkTextActor;
class vtkButtonWidget;
namespace yk::visualization {
class ParameterPanel {
public:
    ParameterPanel(vtkRenderer*, vtkRenderWindowInteractor*, std::function<void()>);
    void configure(const std::vector<EditableParameter>&);
    std::vector<double> values() const;
    bool set(std::size_t, double);
    void enable(bool);
    void status(const std::string&);
    void apply();

private:
    struct Entry {
        EditableParameter spec;
        int group = 0;
        vtkSmartPointer<vtkSliderRepresentation2D> rep;
        vtkSmartPointer<vtkSliderWidget> widget;
        vtkSmartPointer<vtkTextActor> label, value;
    };
    vtkRenderer* renderer_;
    vtkRenderWindowInteractor* interactor_;
    std::function<void()> recompute_;
    std::vector<Entry> entries_;
    vtkSmartPointer<vtkTextActor> status_, button_label_, heading_;
    std::vector<vtkSmartPointer<vtkTextActor>> tabs_;
    vtkSmartPointer<vtkButtonWidget> button_;
    bool enabled_ = false;
    int group_ = 0;
    void showGroup(int);
    void updateValues();
};
}
