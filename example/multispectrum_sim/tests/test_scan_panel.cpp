#include "ui/ScanPanel.hpp"
#include <imgui_internal.h>
#include <iostream>
#include <stdexcept>

namespace yk::spectral {
void testScanPanelInteraction(){
    auto* context=ImGui::CreateContext();
    try{
        auto& io=ImGui::GetIO();io.IniFilename=nullptr;io.DisplaySize={1000,800};io.DeltaTime=1.f/60;
        unsigned char* pixels;int w,h;io.Fonts->GetTexDataAsRGBA32(&pixels,&w,&h);
        SimulationConfig c;c.reconstruction.type="analytic";
        bool changed=false;ImVec2 lo,hi;
        auto frame=[&]{ImGui::NewFrame();ImGui::SetNextWindowPos({0,0});ImGui::SetNextWindowSize({800,600});
            ImGui::Begin("Scan test",nullptr,ImGuiWindowFlags_NoSavedSettings);
            ImGui::SetNextItemWidth(300);changed=ui::ScanPanel::scanTypeCombo(c);
            lo=ImGui::GetItemRectMin();hi=ImGui::GetItemRectMax();ImGui::End();ImGui::Render();};
        auto click=[&](ImVec2 p){io.AddMousePosEvent(p.x,p.y);frame();
            io.AddMouseButtonEvent(0,true);frame();const bool on_press=changed;
            io.AddMouseButtonEvent(0,false);frame();return on_press||changed;};
        const GeometryKind kinds[]={GeometryKind::FlatCbct,GeometryKind::FlatHelical,GeometryKind::PlanarSourceOrbit,GeometryKind::NikonTilted};
        for(int from=0;from<4;++from)for(int to=0;to<4;++to){
            if(from==to)continue;
            c=SimulationConfig{};c.geometry.kind=kinds[from];c.reconstruction.type="analytic";
            c.reconstruction.analytic.pipeline=from==1?"wfbp":"fdk";
            c.geometry.parameters.pitch_mm_per_turn=80;c.geometry.parameters.views=720;c.geometry.parameters.views_per_turn=360;
            frame();frame();
            click({lo.x+20,(lo.y+hi.y)*.5f});
            if(context->OpenPopupStack.empty())throw std::runtime_error("Scan combo did not open");
            frame();auto* popup=context->OpenPopupStack.back().Window;
            if(!popup)throw std::runtime_error("Scan popup missing");
            const auto start=popup->DC.CursorStartPos;
            const ImVec2 item{start.x+60,start.y+to*ImGui::GetTextLineHeightWithSpacing()+ImGui::GetTextLineHeight()*.5f};
            if(!click(item) || c.geometry.kind!=kinds[to])throw std::runtime_error("Scan combo mouse transition failed: "+std::to_string(from)+" -> "+std::to_string(to));
            if(c.reconstruction.analytic.pipeline!=(to==1?"wfbp":"fdk"))throw std::runtime_error("Scan combo left incompatible analytic method");
            frame();if(changed)throw std::runtime_error("Scan combo repeatedly marked unchanged input dirty");
        }
        ImGui::DestroyContext(context);std::cout<<"All 12 scan combo mouse transitions passed\n";
    }catch(...){ImGui::DestroyContext(context);throw;}
}
}
