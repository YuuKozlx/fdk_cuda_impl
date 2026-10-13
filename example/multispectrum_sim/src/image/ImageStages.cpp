#include "image/ImageStages.hpp"
#include <stdexcept>
namespace yk::spectral::image {
namespace {
    class ProjectionStage final : public preprocess::IStage {
        Options options_;
        preprocess::Domain domain_;
        Location location_;
        Workspace workspace_;
        std::unique_ptr<image::IStage> stage_;

    public:
        ProjectionStage(Options o, preprocess::Domain d, Location l)
            : options_(o)
            , domain_(d)
            , location_(l)
            , workspace_(l)
            , stage_(makeStage(o, Domain::Projection, l))
        {
            if (o.operation == Operation::Resample)
                throw std::invalid_argument("resampling changes detector geometry: use image::Pipeline and propagate its output grid");
        }
        const pipeline::PluginInfo& info() const noexcept override { return stage_->info(); }
        pipeline::Contract<preprocess::Domain> contract() const override { return { domain_, domain_, location_, location_ }; }
        void process(preprocess::Frame& f, const preprocess::Context& c) override
        {
            Image image;
            image.grid.dimensions = { f.width, f.height, 1 };
            image.domain = Domain::Projection;
            image.location = location_;
            image.grid.spacing_mm = { c.config.geometry.parameters.pixel_u_mm, c.config.geometry.parameters.pixel_v_mm, 1 };
            if (f.pixels != image.elements() || (location_ == Location::Host && (!f.host_data || f.host_data->size() != f.pixels)))
                throw std::invalid_argument("invalid projection image storage");
            image.data = location_ == Location::Device ? f.device_data : f.host_data->data();
            float* original = image.data;
            auto& workspace = c.image_workspace && c.image_workspace->location() == location_ ? *c.image_workspace : workspace_;
            image::process(image, options_, workspace);
            copy(original, location_, image.data, location_, f.pixels);
            synchronize(location_);
        }
        void reset() noexcept override
        {
            try {
                workspace_.trim();
            } catch (...) {
            }
        }
    };
    class VolumeStage final : public postprocess::IStage {
        Options options_;
        postprocess::Domain domain_;
        Location location_;
        Workspace workspace_;
        std::unique_ptr<image::IStage> stage_;

    public:
        VolumeStage(Options o, postprocess::Domain d, Location l)
            : options_(o)
            , domain_(d)
            , location_(l)
            , workspace_(l)
            , stage_(makeStage(o, Domain::Volume, l))
        {
            if(d==postprocess::Domain::Labels && (o.operation!=Operation::Resample || o.interpolation!=Interpolation::Nearest))
                throw std::invalid_argument("label volumes only support nearest-neighbor resampling");
        }
        const pipeline::PluginInfo& info() const noexcept override { return stage_->info(); }
        pipeline::Contract<postprocess::Domain> contract() const override { return { domain_, domain_, location_, location_ }; }
        void validateContext(const postprocess::Context& c) const override
        {
            if (c.stream)
                throw std::invalid_argument("image stages currently require the pool's default CUDA stream");
        }
        void process(postprocess::Volume& v, const postprocess::Context& c) override
        {
            Image image;
            image.grid = v.grid;
            image.location = location_;
            image.data = location_ == Location::Device ? v.device_data : v.host_data;
            image.owner = location_ == Location::Device ? v.device_owner : v.host_owner;
            if (v.elements != image.elements())
                throw std::invalid_argument("invalid volume image storage");
            auto& workspace = c.image_workspace && c.image_workspace->location() == location_ ? *c.image_workspace : workspace_;
            image::process(image, options_, workspace);
            v.grid = image.grid;
            v.elements = image.elements();
            if (location_ == Location::Device) {
                v.device_owner = image.owner;
                v.device_data = image.data;
                v.host_owner.reset();
                v.host_data = nullptr;
            } else {
                v.host_owner = image.owner;
                v.host_data = image.data;
                v.device_owner.reset();
                v.device_data = nullptr;
            }
        }
        void reset() noexcept override
        {
            try {
                workspace_.trim();
            } catch (...) {
            }
        }
    };
}
std::unique_ptr<preprocess::IStage> makeProjectionStage(Options o, preprocess::Domain d, Location l) { return std::make_unique<ProjectionStage>(o, d, l); }
std::unique_ptr<postprocess::IStage> makeVolumeStage(Options o, postprocess::Domain d, Location l) { return std::make_unique<VolumeStage>(o, d, l); }
void VolumeTransfer::ensure(postprocess::Volume& v, Location l, const postprocess::Context& c)
{
    if (c.stream)
        throw std::invalid_argument("volume transfer requires default CUDA stream");
    if (v.location == l)
        return;
    Image image;
    image.grid = v.grid;
    image.location = v.location;
    image.data = v.location == Location::Device ? v.device_data : v.host_data;
    if (v.elements != image.elements())
        throw std::invalid_argument("volume transfer shape mismatch");
    float* existing = l == Location::Device ? v.device_data : v.host_data;
    if (existing) {
        copy(existing, l, image.data, v.location, v.elements);
        v.location = l;
        if (l == Location::Device)
            ++uploads;
        else
            ++downloads;
        return;
    }
    ensureLocation(image, l);
    if (l == Location::Device) {
        v.device_owner = image.owner;
        v.device_data = image.data;
        ++uploads;
    } else {
        v.host_owner = image.owner;
        v.host_data = image.data;
        ++downloads;
    }
    v.location = l;
}
} // namespace
