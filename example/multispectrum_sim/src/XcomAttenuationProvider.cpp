#include "XcomAttenuationProvider.hpp"
#include "xcom_c_api.h"
#include <algorithm>
#include <cctype>
#include <iterator>
#include <sstream>
#include <string>
namespace yk::spectral {
namespace {
// ICRU/Gate 常用组织。质量分数按原始资料保留，零质量分数不写入。
const BuiltinMaterialInfo builtin[] = {
#if 0
    {"element.aluminum", "element", "Aluminum", "al,aluminium,aluminum", 2.70, "13:1"},
    {"element.copper", "element", "Copper", "cu,copper", 8.96, "29:1"},
    {"element.iron", "element", "Iron", "fe,iron", 7.874, "26:1"},
    {"element.iodine", "element", "Iodine", "i,iodine", 4.93, "53:1"},
    {"element.tungsten", "element", "Tungsten", "w,tungsten", 19.3, "74:1"},
    {"element.lead", "element", "Lead", "pb,lead", 11.4, "82:1"},
    {"compound.water", "compound", "Water", "water,h2o", 1.00, "1:0.111902,8:0.888098"},
    {"mixture.air", "mixture", "Air", "air", 0.00129, "6:0.000124,7:0.755268,8:0.231781,18:0.012827"},
    {"tissue.brain", "human_tissue", "Brain", "brain,icru_brain_adult", 1.04, "1:0.107,6:0.145,7:0.022,8:0.712,11:0.002,15:0.004,16:0.002,17:0.003,19:0.003"},
    {"tissue.muscle", "human_tissue", "Muscle", "muscle,icru_muscle_adult", 1.05, "1:0.102,6:0.143,7:0.034,8:0.710,11:0.001,15:0.002,16:0.003,17:0.001,19:0.004"},
    {"tissue.breast", "human_tissue", "Breast", "breast,icru_breast_adult2", 1.020, "1:0.106,6:0.332,7:0.030,8:0.527,11:0.001,15:0.001,16:0.002,17:0.001"},
    {"tissue.adipose", "human_tissue", "Adipose", "adipose,icru_adipose_adult2", 0.92, "1:0.120,6:0.640,7:0.008,8:0.229,15:0.002,20:0.001"},
    {"tissue.lung", "human_tissue", "Lung", "lung,icru_lung_adult_healthy", 0.26, "1:0.103,6:0.105,7:0.031,8:0.749,11:0.002,15:0.002,16:0.003,17:0.003,19:0.002"},
    {"tissue.lung_moby", "human_tissue", "Lung MOBY", "lungmoby,lung_moby", 0.30, "1:0.099,6:0.100,7:0.028,8:0.740,15:0.001,20:0.032"},
    {"tissue.blood", "human_tissue", "Blood", "blood,icru_blood_adult", 1.06, "1:0.102,6:0.110,7:0.033,8:0.745,11:0.001,15:0.001,16:0.002,17:0.003,19:0.002,26:0.001"},
    {"tissue.heart", "human_tissue", "Heart", "heart", 1.05, "1:0.104,6:0.139,7:0.029,8:0.718,11:0.001,15:0.002,16:0.002,17:0.002,19:0.003"},
    {"tissue.kidney", "human_tissue", "Kidney", "kidney,icru_kidney_adult", 1.05, "1:0.103,6:0.132,7:0.030,8:0.724,11:0.002,15:0.002,16:0.002,17:0.002,19:0.002,20:0.001"},
    {"tissue.liver", "human_tissue", "Liver", "liver,icru_liver_adult", 1.06, "1:0.102,6:0.139,7:0.030,8:0.716,11:0.002,15:0.003,16:0.003,17:0.002,19:0.003"},
    {"tissue.pancreas", "human_tissue", "Pancreas", "pancreas,icru_pancreas_adult", 1.04, "1:0.106,6:0.169,7:0.022,8:0.694,11:0.002,15:0.002,16:0.001,17:0.002,19:0.002"},
    {"tissue.spleen", "human_tissue", "Spleen", "spleen,icru_spleen_adult", 1.06, "1:0.103,6:0.113,7:0.032,8:0.741,11:0.001,15:0.003,16:0.002,17:0.002,19:0.003"},
    {"tissue.cartilage", "human_tissue", "Cartilage", "cartilage", 1.10, "1:0.096,6:0.099,7:0.022,8:0.744,11:0.005,15:0.022,16:0.009,17:0.003"},
    {"tissue.cortical_bone", "human_tissue", "Cortical bone", "bone,corticalbone,icru_skeleton_cortical_bone_adult", 1.92, "1:0.034,6:0.155,7:0.042,8:0.435,11:0.001,12:0.002,15:0.103,16:0.003,20:0.225"},
    {"tissue.spine_bone", "human_tissue", "Spine bone", "spinebone,spine_bone", 1.42, "1:0.063,6:0.261,7:0.039,8:0.436,11:0.001,12:0.001,15:0.061,16:0.003,17:0.001,19:0.001,20:0.133"},
    {"tissue.skull", "human_tissue", "Skull", "skull", 1.61, "1:0.050,6:0.212,7:0.040,8:0.435,11:0.001,12:0.002,15:0.081,16:0.003,20:0.176"},
 #endif
    {"", "", "", "", 0.0, ""}
};

const BuiltinMaterialInfo generated_builtin[] = {
#include "BuiltinMaterials.inc"
#include "BuiltinGateMaterials.inc"
};

std::string key(std::string_view value)
{
    std::string result(value);
    result.erase(std::remove_if(result.begin(), result.end(),
        [](unsigned char c) { return std::isspace(c) || c == '_' || c == '-'; }),
        result.end());
    std::transform(result.begin(), result.end(), result.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

void parseComposition(const char* text, std::vector<int>& z,
    std::vector<double>& fractions)
{
    std::istringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        const auto split = item.find(':');
        if (split == std::string::npos) continue;
        z.push_back(std::stoi(item.substr(0, split)));
        fractions.push_back(std::stod(item.substr(split + 1)));
    }
}
}

const BuiltinMaterialInfo* findBuiltinMaterial(std::string_view name)
{
    const auto wanted = key(name);
    for (const auto& material : builtin) {
        if (material.id[0] == '\0') continue;
        if (wanted == key(material.id)) return &material;
        std::istringstream aliases(material.aliases);
        std::string alias;
        while (std::getline(aliases, alias, ','))
            if (wanted == key(alias)) return &material;
    }
    for (const auto& material : generated_builtin) {
        if (wanted == key(material.id)) return &material;
        std::istringstream aliases(material.aliases);
        std::string alias;
        while (std::getline(aliases, alias, ','))
            if (wanted == key(alias)) return &material;
    }
    return nullptr;
}

const BuiltinMaterialInfo* builtinMaterialsBegin() { return std::begin(generated_builtin); }
const BuiltinMaterialInfo* builtinMaterialsEnd() { return std::end(generated_builtin); }

XcomAttenuationProvider::XcomAttenuationProvider(std::filesystem::path p):data_directory_(std::move(p)){}
bool XcomAttenuationProvider::query(const MaterialSpec& m,const std::vector<double>& e,
                                    std::vector<double>& out,std::string& error) const {
    out.clear(); if((m.formula.empty() && m.preset.empty())||e.empty()){error="材料 formula/preset 和能量列表不能为空";return false;}
    XcomResult r{}; char msg[512]{};
    int status = 0;
    if (!m.preset.empty()) {
        const auto* material = findBuiltinMaterial(m.preset);
        if (!material) { error = "未知内置材料 preset: " + m.preset; return false; }
        std::vector<int> z;
        std::vector<double> fractions;
        parseComposition(material->composition, z, fractions);
        status = xcom_calculate_mass_fractions(data_directory_.string().c_str(),
            z.data(), fractions.data(), z.size(), e.data(), e.size(), &r, msg, sizeof(msg));
    } else {
        status = xcom_calculate_formula(data_directory_.string().c_str(), m.formula.c_str(),
            e.data(), e.size(), &r, msg, sizeof(msg));
    }
    if(status!=0){error=msg;return false;}
    // 主射线中发生相干散射的光子同样离开原传播方向，因此采用包含相干散射的总衰减。
    out.assign(r.total_with_coherent,r.total_with_coherent+r.length); xcom_free_result(&r); return true;
}
}
