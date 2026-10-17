#pragma once
#include <imgui.h>
namespace yk::visualization {
inline void applyClientTheme()
{
    ImGui::StyleColorsDark();
    auto& s = ImGui::GetStyle();
    s.WindowPadding = { 18, 16 };
    s.FramePadding = { 10, 7 };
    s.ItemSpacing = { 10, 9 };
    s.ItemInnerSpacing = { 8, 6 };
    s.WindowRounding = 10;
    s.ChildRounding = 8;
    s.FrameRounding = 5;
    s.PopupRounding = 9;
    s.GrabRounding = 5;
    s.TabRounding = 6;
    s.ScrollbarRounding = 8;
    s.ScrollbarSize = 12;
    s.GrabMinSize = 12;
    s.WindowBorderSize = 0;
    s.ChildBorderSize = 0;
    s.FrameBorderSize = 0;
    s.PopupBorderSize = 1;
    auto* c = s.Colors;
    c[ImGuiCol_Text] = { .92f, .93f, .95f, 1 };
    c[ImGuiCol_TextDisabled] = { .59f, .62f, .67f, 1 };
    c[ImGuiCol_WindowBg] = { .115f, .125f, .145f, 1 };
    c[ImGuiCol_ChildBg] = c[ImGuiCol_WindowBg];
    c[ImGuiCol_PopupBg] = { .16f, .17f, .195f, 1 };
    c[ImGuiCol_Border] = { .36f, .38f, .43f, .35f };
    c[ImGuiCol_TitleBg] = c[ImGuiCol_WindowBg];
    c[ImGuiCol_TitleBgActive] = c[ImGuiCol_WindowBg];
    c[ImGuiCol_MenuBarBg] = { .14f, .15f, .175f, 1 };
    c[ImGuiCol_FrameBg] = { .19f, .205f, .235f, 1 };
    c[ImGuiCol_FrameBgHovered] = { .24f, .26f, .30f, 1 };
    c[ImGuiCol_FrameBgActive] = { .28f, .30f, .35f, 1 };
    c[ImGuiCol_Button] = { .25f, .29f, .40f, 1 };
    c[ImGuiCol_ButtonHovered] = { .33f, .39f, .54f, 1 };
    c[ImGuiCol_ButtonActive] = { .38f, .45f, .62f, 1 };
    c[ImGuiCol_Header] = { .18f, .195f, .225f, 1 };
    c[ImGuiCol_HeaderHovered] = { .25f, .275f, .33f, 1 };
    c[ImGuiCol_HeaderActive] = { .28f, .32f, .41f, 1 };
    c[ImGuiCol_CheckMark] = { .59f, .70f, .98f, 1 };
    c[ImGuiCol_SliderGrab] = { .47f, .58f, .82f, 1 };
    c[ImGuiCol_SliderGrabActive] = c[ImGuiCol_CheckMark];
    c[ImGuiCol_Tab] = { .16f, .175f, .20f, 1 };
    c[ImGuiCol_TabHovered] = { .28f, .32f, .42f, 1 };
    c[ImGuiCol_TabSelected] = { .235f, .265f, .34f, 1 };
    c[ImGuiCol_TabSelectedOverline] = c[ImGuiCol_CheckMark];
    c[ImGuiCol_TabDimmed] = c[ImGuiCol_Tab];
    c[ImGuiCol_TabDimmedSelected] = c[ImGuiCol_TabSelected];
    c[ImGuiCol_Separator] = { .33f, .35f, .40f, .45f };
    c[ImGuiCol_ScrollbarBg] = c[ImGuiCol_WindowBg];
    c[ImGuiCol_ScrollbarGrab] = { .31f, .33f, .38f, 1 };
    c[ImGuiCol_ScrollbarGrabHovered] = { .4f, .43f, .49f, 1 };
    c[ImGuiCol_ScrollbarGrabActive] = { .46f, .5f, .58f, 1 };
}
}