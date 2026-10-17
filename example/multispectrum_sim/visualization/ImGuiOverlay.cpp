#include "ImGuiTheme.hpp"
#include <windows.h>

#include "ImGuiOverlay.hpp"
#include <backends/imgui_impl_opengl3.h>
#include <backends/imgui_impl_win32.h>
#include <commctrl.h>
#include <imgui.h>
#include <stdexcept>
#include <vtkObjectFactory.h>
#include <vtkProp.h>
#include <vtkRenderWindow.h>
#include <vtkWin32OpenGLRenderWindow.h>
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);
namespace yk::visualization {
namespace {
    class OverlayProp : public vtkProp {
    public:
        static OverlayProp* New();
        vtkTypeMacro(OverlayProp, vtkProp);
        std::function<void()> draw;
        int RenderOverlay(vtkViewport*) override
        {
            if (draw)
                draw();
            return 1;
        }
    };
    vtkStandardNewMacro(OverlayProp);
}
struct ImGuiOverlay::Impl {
    vtkRenderWindow* window;
    vtkSmartPointer<OverlayProp> actor = vtkSmartPointer<OverlayProp>::New();
    std::function<void()> draw;
    ImGuiContext* context = nullptr;
    HWND hwnd = nullptr;
    bool platform_ready=false, renderer_ready=false;
    static LRESULT CALLBACK messages(HWND h, UINT msg, WPARAM wp, LPARAM lp, UINT_PTR, DWORD_PTR data)
    {
        auto& self = *reinterpret_cast<Impl*>(data);
        if (self.context) {
            ImGui::SetCurrentContext(self.context);
            const LRESULT handled = ImGui_ImplWin32_WndProcHandler(h, msg, wp, lp);
            const auto& io = ImGui::GetIO();
            const bool mouse = msg >= WM_MOUSEFIRST && msg <= WM_MOUSELAST;
            const bool keyboard = (msg >= WM_KEYFIRST && msg <= WM_KEYLAST) || msg == WM_CHAR;
            if (handled || (mouse && io.WantCaptureMouse) || (keyboard && io.WantCaptureKeyboard))
                return 0;
        }
        return DefSubclassProc(h, msg, wp, lp);
    }
    void render()
    {
        if (!context) {
            auto* native = vtkWin32OpenGLRenderWindow::SafeDownCast(window);
            if (!native || !native->GetWindowId())
                throw std::runtime_error("ImGui needs a native Windows VTK window");
            hwnd = static_cast<HWND>(native->GetWindowId());
            context = ImGui::CreateContext();
            ImGui::GetIO().IniFilename = nullptr;
            ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
            ImFontConfig font;
            font.SizePixels=20;
            if (GetFileAttributesW(L"C:/Windows/Fonts/segoeui.ttf") != INVALID_FILE_ATTRIBUTES)
                ImGui::GetIO().Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeui.ttf",20);
            else ImGui::GetIO().Fonts->AddFontDefault(&font);
            yk::visualization::applyClientTheme();
            platform_ready=ImGui_ImplWin32_InitForOpenGL(hwnd);
            renderer_ready=platform_ready && ImGui_ImplOpenGL3_Init("#version 150");
            if(!platform_ready || !renderer_ready) throw std::runtime_error("cannot initialize ImGui backends");
            if (!SetWindowSubclass(hwnd, messages, 1, reinterpret_cast<DWORD_PTR>(this)))
                throw std::runtime_error("cannot attach ImGui input handler");
        }
        ImGui::SetCurrentContext(context);
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        draw();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
    ~Impl()
    {
        if (context) {
            ImGui::SetCurrentContext(context);
            if (IsWindow(hwnd))
                RemoveWindowSubclass(hwnd, messages, 1);
            window->MakeCurrent();
            if(renderer_ready) ImGui_ImplOpenGL3_Shutdown();
            if(platform_ready) ImGui_ImplWin32_Shutdown();
            ImGui::DestroyContext(context);
        }
    }
};
ImGuiOverlay::ImGuiOverlay(vtkRenderWindow* w, std::function<void()> draw)
    : impl_(std::make_unique<Impl>())
{
    impl_->window = w;
    impl_->draw = std::move(draw);
    impl_->actor->draw = [this] { impl_->render(); };
}
ImGuiOverlay::~ImGuiOverlay() { impl_->actor->draw = {}; }
vtkProp* ImGuiOverlay::prop() const { return impl_->actor; }
}
