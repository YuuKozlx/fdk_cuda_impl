#include "DataFidelity/WeightedLeastSquares.hpp"
#include "Optimizer/GradientDescent.hpp"
#include "Optimizer/RegularizedAlgebraic.hpp"
namespace YK::Iter::Statistical {
std::unique_ptr<IDataFidelity> createDataFidelity(EDataFidelityKind method)
{
    if (method == EDataFidelityKind::WeightedLeastSquares)
        return std::make_unique<WeightedLeastSquares>();
    return {};
}
std::unique_ptr<IOptimizer> createOptimizer(EStatisticalOptimizer method)
{
    if(method==EStatisticalOptimizer::RegularizedAlgebraic)return std::make_unique<RegularizedAlgebraic>();
    if (method == EStatisticalOptimizer::GradientDescent)
        return std::make_unique<GradientDescent>();
    return {};
}
}
