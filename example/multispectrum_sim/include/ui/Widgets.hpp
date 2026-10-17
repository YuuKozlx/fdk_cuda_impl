#pragma once
#include <algorithm>
#include <array>
#include <windows.h>
#include <commdlg.h>
#include <filesystem>
#include <imgui.h>
#include <string>

namespace yk::spectral::ui {
inline constexpr double degrees = 57.29577951308232;
inline std::filesystem::path chooseFile(bool save, const wchar_t* filter)
{
    std::array<wchar_t, 32768> buffer {};
    OPENFILENAMEW info {};
    info.lStructSize = sizeof(info);
    info.lpstrFile = buffer.data();
    info.nMaxFile = DWORD(buffer.size());
    info.lpstrFilter = filter;
    info.nFilterIndex = 1;
    info.Flags = OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST | (save ? OFN_OVERWRITEPROMPT : OFN_FILEMUSTEXIST);
    info.hwndOwner = GetActiveWindow();
    if (save ? GetSaveFileNameW(&info) : GetOpenFileNameW(&info))
        return buffer.data();
    return {};
}
inline void stringInput(const char* name, std::string& text)
{
    std::array<char, 2048> b {};
    std::copy_n(text.c_str(), std::min(text.size(), b.size() - 1), b.data());
    ImGui::TextUnformatted(name);
    ImGui::SetNextItemWidth(-1);
    const std::string id = std::string("##") + name;
    if (ImGui::InputText(id.c_str(), b.data(), b.size()))
        text = b.data();
}
inline void pathInput(const char* name, std::filesystem::path& path, const wchar_t* filter)
{
    ImGui::PushID(name);
    std::string s = path.generic_string();
    stringInput(name, s);
    path = s;
    if (ImGui::Button("Browse...")) {
        const auto selected = chooseFile(false, filter);
        if (!selected.empty())
            path = selected;
    }
    ImGui::PopID();
}
inline std::string shortLabel(std::string text, float width)
{
    if (ImGui::CalcTextSize(text.c_str()).x <= width)
        return text;
    while (!text.empty() && ImGui::CalcTextSize((text + "...").c_str()).x > width) {
        std::size_t start = text.size() - 1;
        while (start > 0 && (static_cast<unsigned char>(text[start]) & 0xc0) == 0x80)
            --start;
        text.resize(start);
    }
    return text + "...";
}
inline bool beginField(const char* label)
{
    ImGui::PushID(label);
    if (!ImGui::BeginTable("field", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::PopID();
        return false;
    }
    ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthStretch, .52f);
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, .48f);
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::TextWrapped("%s", label);
    ImGui::TableNextColumn();
    ImGui::SetNextItemWidth(-1);
    return true;
}
inline void endField()
{
    ImGui::EndTable();
    ImGui::PopID();
}
inline void number(const char* label, double& value)
{
    if (beginField(label)) {
        ImGui::InputDouble("##value", &value, 0, 0, "%.6g");
        endField();
    }
}
inline void integer(const char* label, int& value)
{
    if (beginField(label)) {
        ImGui::InputInt("##value", &value);
        endField();
    }
}
inline void angle(const char* label, double& value)
{
    double d = value * degrees;
    number(label, d);
    value = d / degrees;
}

}