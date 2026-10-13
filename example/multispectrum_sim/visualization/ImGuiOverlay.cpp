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
            ImGui::StyleColorsDark();
            auto& style=ImGui::GetStyle();
            style.WindowPadding={14,12};style.FramePadding={10,6};style.ItemSpacing={8,8};style.ItemInnerSpacing={8,6};
            style.WindowRounding=8;style.ChildRounding=6;style.FrameRounding=6;style.PopupRounding=8;style.GrabRounding=6;style.TabRounding=6;
            style.ScrollbarRounding=8;style.ScrollbarSize=13;style.GrabMinSize=12;
            style.WindowBorderSize=0;style.ChildBorderSize=0;style.FrameBorderSize=0;style.PopupBorderSize=1;
            auto* color=style.Colors;
            color[ImGuiCol_Text]={.90f,.93f,.97f,1};color[ImGuiCol_TextDisabled]={.48f,.55f,.65f,1};
            color[ImGuiCol_WindowBg]={.065f,.09f,.135f,1};color[ImGuiCol_ChildBg]={.065f,.09f,.135f,1};color[ImGuiCol_PopupBg]={.085f,.12f,.175f,1};
            color[ImGuiCol_Border]={.22f,.29f,.39f,.55f};
            color[ImGuiCol_TitleBg]={.065f,.09f,.135f,1};color[ImGuiCol_TitleBgActive]=color[ImGuiCol_TitleBg];
            color[ImGuiCol_FrameBg]={.115f,.16f,.225f,1};color[ImGuiCol_FrameBgHovered]={.16f,.225f,.31f,1};color[ImGuiCol_FrameBgActive]={.18f,.27f,.37f,1};
            color[ImGuiCol_Button]={.12f,.23f,.34f,1};color[ImGuiCol_ButtonHovered]={.16f,.34f,.46f,1};color[ImGuiCol_ButtonActive]={.15f,.42f,.53f,1};
            color[ImGuiCol_Header]={.12f,.18f,.265f,1};color[ImGuiCol_HeaderHovered]={.15f,.26f,.36f,1};color[ImGuiCol_HeaderActive]={.15f,.33f,.43f,1};
            color[ImGuiCol_CheckMark]={.34f,.79f,.86f,1};color[ImGuiCol_SliderGrab]={.29f,.68f,.77f,1};color[ImGuiCol_SliderGrabActive]={.42f,.87f,.93f,1};
            color[ImGuiCol_Tab]={.085f,.125f,.19f,1};color[ImGuiCol_TabHovered]={.15f,.30f,.41f,1};
            color[ImGuiCol_TabSelected]={.12f,.23f,.33f,1};color[ImGuiCol_TabSelectedOverline]={.34f,.79f,.86f,1};
            color[ImGuiCol_TabDimmed]=color[ImGuiCol_Tab];color[ImGuiCol_TabDimmedSelected]=color[ImGuiCol_TabSelected];
            color[ImGuiCol_Separator]={.22f,.29f,.39f,.6f};
            color[ImGuiCol_ScrollbarBg]={.065f,.09f,.135f,1};color[ImGuiCol_ScrollbarGrab]={.23f,.30f,.4f,1};
            color[ImGuiCol_ScrollbarGrabHovered]={.3f,.4f,.51f,1};color[ImGuiCol_ScrollbarGrabActive]={.34f,.5f,.62f,1};
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
