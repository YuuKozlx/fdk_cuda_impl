#pragma once
#include "input/config/SimTypes.hpp"
#include <memory>
namespace yk::spectral::preprocess {
class CudaPixelOps {
  public:
    explicit CudaPixelOps(size_t);
    ~CudaPixelOps();
    void beginFrame();
    void upload(const std::vector<float> &log, const std::vector<float> *air);
    void download(std::vector<float> &);
    void downloadFrom(const float *, std::vector<float> &);
    void uploadTo(float *, const std::vector<float> &);
    float *data();
    float *air();
    float *original();
    float *guide();
    float *guideAir();
    float *prediction();
    void restoreEnergy();
    void convertTransmission(bool to_transmission, bool energy_domain = false);
    const unsigned char* makeMaskDevice(float*,const float*,double);
    std::vector<unsigned char> mask(float *energy, const float *air, double threshold);
    void guidedResidual(double scale);
    void configureBhc(const BeamHardeningConfig &);
    void finalize(bool energy_domain, bool bhc);
    void synchronize();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace yk::spectral::preprocess
