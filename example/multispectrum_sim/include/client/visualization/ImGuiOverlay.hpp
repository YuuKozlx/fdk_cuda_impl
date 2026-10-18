#pragma once
#include <functional>
#include <memory>
#include <vtkSmartPointer.h>
class vtkProp;
class vtkRenderWindow;
namespace yk::visualization {
class ImGuiOverlay {
public:
    ImGuiOverlay(vtkRenderWindow*, std::function<void()>);
    ~ImGuiOverlay();
    vtkProp* prop() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
