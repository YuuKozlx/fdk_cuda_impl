#include "XcomAttenuationProvider.hpp"
#include "xcom_c_api.h"
namespace yk::spectral {
XcomAttenuationProvider::XcomAttenuationProvider(std::filesystem::path p):data_directory_(std::move(p)){}
bool XcomAttenuationProvider::query(const MaterialSpec& m,const std::vector<double>& e,
                                    std::vector<double>& out,std::string& error) const {
    out.clear(); if(m.formula.empty()||e.empty()){error="材料化学式和能量列表不能为空";return false;}
    XcomResult r{}; char msg[512]{};
    if(xcom_calculate_formula(data_directory_.string().c_str(),m.formula.c_str(),e.data(),e.size(),&r,msg,sizeof(msg))!=0){error=msg;return false;}
    // 主射线中发生相干散射的光子同样离开原传播方向，因此采用包含相干散射的总衰减。
    out.assign(r.total_with_coherent,r.total_with_coherent+r.length); xcom_free_result(&r); return true;
}
}
