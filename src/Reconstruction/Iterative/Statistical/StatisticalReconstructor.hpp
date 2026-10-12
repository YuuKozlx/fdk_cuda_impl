#pragma once
#include "Reconstruction/Iterative/Common/YkReconstructionMemoryBudget.hpp"
#include "StatisticalContracts.hpp"
#include "DataFidelity/ProjectionInputAdapter.hpp"
#include "global/YkLog.h"
namespace YK::Iter::Statistical {
class StatisticalReconstructor {
    std::unique_ptr<IDataFidelity> data_;
    std::unique_ptr<IOptimizer> optimizer_;
    std::unique_ptr<IAlgebraicRegularizer> regularizer_;
    SStatisticalReconstructionSpec config_ {};
    std::string plugin_id_, error_;
    bool prepared_ = false;

public:
    ~StatisticalReconstructor() { release(); }
    bool prepare(const SReconstructionParams& p,const std::vector<SConeProjGeomVec>& geometry,const SStatisticalReconstructionSpec& cfg,
        ETask fp,ETask bp,cudaStream_t stream,int device){
        return prepare(p,geometry,cfg,fp,bp,stream,device,defaultComposition(cfg));
    }
    // Internal-only override for strategy comparisons.
    bool prepare(const SReconstructionParams& p,const std::vector<SConeProjGeomVec>& geometry,const SStatisticalReconstructionSpec& cfg,
        ETask fp,ETask bp,cudaStream_t stream,int device,const SRegularizationCompositionSpec& composition){
        try{return prepareImpl(p,geometry,cfg,fp,bp,stream,device,composition);}
        catch(const std::exception& e){error_=e.what();release();return false;}
        catch(...){error_="statistical preparation threw an exception";release();return false;}
    }
    bool prepareImpl(const SReconstructionParams& p, const std::vector<SConeProjGeomVec>& geometry, const SStatisticalReconstructionSpec& cfg,
        ETask fp, ETask bp, cudaStream_t stream, int device, const SRegularizationCompositionSpec& composition)
    {
        release();
        error_.clear();
        config_ = cfg;
        plugin_id_ = cfg.regularization.plugin_id ? cfg.regularization.plugin_id : "none";
        config_.regularization.plugin_id = nullptr;
        if (!std::isfinite(cfg.regularization.strength) || cfg.regularization.strength < 0)
            return false;
        optimizer_ = createOptimizer(cfg.optimizer.kind);
        if (!optimizer_ || !optimizer_->configureComposition(composition)) {
            error_ = "requested statistical optimizer is not implemented";
            return false;
        }
        data_ = createDataFidelity(cfg.data_fidelity.kind);
        if (!data_) {
            error_ = "requested data fidelity is not implemented";
            return false;
        }
        const auto required = optimizer_->requirements();
        const auto dc = data_->capabilities();
        if ((required.data_algebraic_update&&!dc.algebraic_update)||(required.data_value && !dc.value) || (required.data_gradient && !dc.gradient) || ((required.verified_adjoint || cfg.data_fidelity.require_verified_adjoint) && dc.adjoint != AdjointStatus::Verified)) {
            error_ = "data fidelity/operator capabilities do not satisfy optimizer";
            return false;
        }
        AlgebraicRegularizationConfig reg_config;
        size_t regularization_bytes = 0;
        if (plugin_id_ != "none" && !plugin_id_.empty()) {
            regularizer_ = createRegularizerPlugin(plugin_id_);
            if (!regularizer_) {
                error_ = "unknown statistical regularizer";
                return false;
            }
            auto& c = reg_config;
            c.plugin_id = plugin_id_;
            c.strength = cfg.regularization.strength;
            c.huber_delta = c.epsilon = cfg.regularization.shape;
            c.tv_dimensionality = static_cast<ETvDimensionality>(cfg.regularization.dimensions);
            c.gradient_only = !required.regularization_prox;
            c.require_prox = required.regularization_prox;
            c.inner_iterations = composition.prox_iterations;
            c.step_size = composition.prox_step;
            c.lower_bound = cfg.constraints.lower_bound;
            c.upper_bound = cfg.constraints.upper_bound;
            c.non_negative = cfg.constraints.lower_bound >= 0;
            const auto rc = regularizer_->capabilities();
            if ((required.regularization_value && !rc.value) || (required.regularization_gradient && !rc.gradient) || (required.regularization_prox && !rc.prox)) {
                error_ = "regularizer capabilities do not satisfy optimizer";
                return false;
            }
            regularization_bytes = regularizer_->workspaceBytes(p, c);
        } else if (cfg.regularization.strength > 0) {
            error_ = "positive strength requires a regularizer";
            return false;
        }
        const size_t total_bytes = optimizer_->workspaceBytes(size_t(p.volume.Nx) * p.volume.Ny * p.volume.Nz) + data_->workspaceBytes(p) + (required.data_algebraic_update?data_->algebraicWorkspaceBytes(p):0) + regularization_bytes;
        if (!detail::fitsReconstructionBudget(total_bytes, "statistical reconstruction total workspace"))
            return false;
        if (regularizer_ && !regularizer_->prepare(p, reg_config, stream, device)) {
            error_ = "invalid statistical regularizer configuration";
            return false;
        }
        if (!data_->prepare({ p, geometry, cfg.data_fidelity, fp, bp, stream, device, required.data_algebraic_update }))
            return false;
        prepared_ = optimizer_->prepare(size_t(p.volume.Nx) * p.volume.Ny * p.volume.Nz, cfg.optimizer, cfg.constraints, stream, device);
        if (!prepared_)
            release();
        return prepared_;
    }
    bool reconstruct(const float* measured,float* volume,detail::ProjectionInput* source,const float* weights=nullptr,detail::ProjectionInput* weight_source=nullptr){
        try{return reconstructImpl(measured,volume,source,weights,weight_source);}
        catch(const std::exception& e){error_=e.what();return false;}
        catch(...){error_="statistical execution threw an exception";return false;}
    }
    bool reconstructImpl(const float* measured, float* volume, detail::ProjectionInput* source, const float* weights = nullptr, detail::ProjectionInput* weight_source = nullptr)
    {
        if (!prepared_)
            return false;
        error_.clear();
        std::unique_ptr<ProjectionInputAdapter> projection_adapter,weight_adapter;
        if(source)projection_adapter=std::make_unique<ProjectionInputAdapter>(*source);
        if(weight_source)weight_adapter=std::make_unique<ProjectionInputAdapter>(*weight_source);
        struct UnbindOnExit { IDataFidelity& data; ~UnbindOnExit(){data.unbind();} } unbind{*data_};
        if (!data_->bind({ measured, projection_adapter.get(), weights, weight_adapter.get() })) {
            error_ = data_->error();
            return false;
        }
        ReconstructionProblem problem { *data_, regularizer_.get(), config_.regularization.strength, regularizer_ ? regularizer_->gradientLipschitz() : 0 };
        if (!optimizer_->run(problem, volume)) {
            error_ = optimizer_->error();
            YK_LOGE("[Statistical] {}", error_);
            return false;
        }
        const auto& s = optimizer_->statistics();
        YK_LOGI("[Statistical] iterations={} objective={:.7e}->{:.7e} step={:.7e} backtracking={} converged={} adjoint={}",
            s.completed_iterations, s.initial_objective, s.final_objective, s.last_step, s.total_backtracking, s.converged, data_->capabilities().adjoint == AdjointStatus::Verified ? "verified" : data_->capabilities().adjoint == AdjointStatus::Approximate ? "approximate"
                                                                                                                                                                                                                                                                : "unverified");
        return true;
    }
    const std::string& error() const { return error_; }
    void reset()
    {
        if (optimizer_)
            optimizer_->reset();
        if (regularizer_)
            regularizer_->reset();
        error_.clear();
    }
    void release()
    {
        if (optimizer_)
            optimizer_->release();
        optimizer_.reset();
        if (regularizer_)
            regularizer_->release();
        regularizer_.reset();
        data_.reset();
        prepared_ = false;
    }
};
}
