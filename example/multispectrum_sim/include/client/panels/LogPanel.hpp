#pragma once
#include "processing/support/SimLogger.hpp"
#include <imgui.h>
#include <deque>
#include <sstream>
namespace yk::spectral::ui {
// A bounded, read-only view. Workers only write SimLogger; ImGui stays on the UI thread.
struct LogPanel {
    bool open=false,follow=true;
    std::uint64_t sequence=0;
    std::string last_status;
    std::deque<std::string> lines;
    ImGuiTextFilter filter;
    void append(const std::string& text){
        std::istringstream input(text.substr(0,4096));std::string line;
        while(std::getline(input,line)){lines.push_back(std::move(line));if(lines.size()>2000)lines.pop_front();}
    }
    void draw(const std::string& status,int width,int height){
        bool changed=false;
        for(const auto& entry:SimLogger::instance().readSince(sequence)){
            sequence=entry.first;append(entry.second);changed=true;
        }
        if(status!=last_status){last_status=status;if(!status.empty()){append("[Status] "+status);changed=true;}}
        if(!open)return;
        ImGui::SetNextWindowSize({float(width)*.7f,float(height)*.38f},ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos({float(width)*.3f,float(height)*.6f},ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Runtime log",&open)){
            if(ImGui::Button("Clear")){lines.clear();changed=true;}
            ImGui::SameLine();
            if(ImGui::Button("Copy visible")){
                std::string text;for(const auto& line:lines)if(filter.PassFilter(line.c_str())){text+=line;text+='\n';}
                ImGui::SetClipboardText(text.c_str());
            }
            ImGui::SameLine();ImGui::Checkbox("Follow latest",&follow);
            filter.Draw("Filter",-1);
            ImGui::TextDisabled("Application logs and task status | recent 2000 lines | read-only");
            ImGui::BeginChild("Log lines",{0,0},true,ImGuiWindowFlags_HorizontalScrollbar);
            std::vector<const std::string*> visible;
            for(const auto& line:lines)if(filter.PassFilter(line.c_str()))visible.push_back(&line);
            ImGuiListClipper clipper;clipper.Begin(int(visible.size()));
            while(clipper.Step())for(int i=clipper.DisplayStart;i<clipper.DisplayEnd;++i)ImGui::TextUnformatted(visible[i]->c_str());
            if(follow && changed)ImGui::SetScrollHereY(1.f);
            ImGui::EndChild();
        }
        ImGui::End();
    }
};
}
