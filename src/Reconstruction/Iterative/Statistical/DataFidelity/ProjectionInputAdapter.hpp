#pragma once
#include "Reconstruction/Iterative/Statistical/StatisticalContracts.hpp"
#include "Reconstruction/Iterative/Common/YkProjectionInput.hpp"
namespace YK::Iter::Statistical {
class ProjectionInputAdapter final:public IProjectionDataSource {
    detail::ProjectionInput& input_;
public:
    explicit ProjectionInputAdapter(detail::ProjectionInput& input):input_(input){}
    int preferredViews()const override{return input_.preferredViews();}
    bool prepare(int count,cudaStream_t stream)override{return input_.prepare(count,stream);}
    const float* fetch(const std::vector<int>& views,cudaStream_t stream)override{return input_.fetch(views,stream);}
    const std::string& error()const override{return input_.error();}
};
}
