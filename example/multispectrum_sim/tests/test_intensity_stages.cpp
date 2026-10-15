#include "recon/postprocess/stages/IntensityPluginInfo.hpp"
#include "recon/postprocess/stages/correction/HuTransforms.hpp"
#include "recon/postprocess/stages/segmentation/HuThresholds.hpp"
#ifdef YK_INTENSITY_CUDA
#include "image/ImageStages.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace yk::spectral;
using namespace yk::spectral::postprocess;
void require(bool b, const char* what)
{
    if (!b)
        throw std::runtime_error(what);
}
template <class F>
void rejects(F f)
{
    bool caught = false;
    try {
        f();
    } catch (const std::exception&) {
        caught = true;
    }
    require(caught, "invalid intensity request accepted");
}
struct Data {
    std::vector<float> host;
    Volume volume;
    Context context;
#ifdef YK_INTENSITY_CUDA
    image::VolumeTransfer transfer;
#endif
    explicit Data(std::vector<float> values, Domain domain)
        : host(std::move(values))
    {
        volume.grid.dimensions = { int(host.size()), 1, 1 };
        volume.elements = host.size();
        volume.host_data = host.data();
        volume.domain = domain;
#ifdef YK_INTENSITY_CUDA
        context.transfer = &transfer;
#endif
    }
    void download()
    {
#ifdef YK_INTENSITY_CUDA
        transfer.ensure(volume, Location::Host, context);
#endif
    }
};
void test(Location location)
{
    // Independent affine oracle over configurable reference anchors and values
    // spanning air, water, bone and values outside the calibrated interval.
    for (double waterMu : { .01, .02, .04 })
        for (double airMu : { 0., .0001 })
            for (double waterHu : { 0., 30. }) {
                MuHuReference ref { waterMu, airMu, waterHu, -1000 };
                std::vector<float> source;
                for (int k = 0; k < 513; ++k)
                    source.push_back(float(airMu + (waterMu - airMu) * (-.2 + 3.4 * k / 512)));
                Data d(source, Domain::AttenuationMmInv);
                Pipeline toHu;
                toHu.append(makeMuToHuStage(ref, location));
                toHu.run(d.volume, d.context);
                d.download();
                for (size_t k = 0; k < source.size(); ++k) {
                    double expected = -1000 + (source[k] - airMu) * (waterHu + 1000) / (waterMu - airMu);
                    require(std::abs(d.volume.host_data[k] - expected) < .0005, "reference matrix forward mismatch");
                }
                Pipeline toMu;
                toMu.append(makeHuToMuStage(ref, location));
                toMu.run(d.volume, d.context);
                d.download();
                for (size_t k = 0; k < source.size(); ++k)
                    require(std::abs(d.volume.host_data[k] - source[k]) < 2e-8, "reference matrix roundtrip mismatch");
            }
    for (double air : { -1200., -900., -500. })
        for (double water : { -100., 0., 200. }) {
            auto fit = fitAirWaterHu(air, water);
            Data d({ float(air), float(water), float((air + water) / 2) }, Domain::HounsfieldUnits);
            Pipeline p;
            p.append(makeHuCorrectionStage(fit.coefficients, location));
            p.run(d.volume, d.context);
            d.download();
            for (int k = 0; k < 3; ++k)
                require(std::abs(d.volume.host_data[k] - (k == 0 ? -1000 : k == 1 ? 0
                                                                                  : -500))
                        < .0002,
                    "calibration matrix anchor mismatch");
        }
    const MuHuReference reference { .025, .0001, 20, -980 };
    std::vector<float> original = { -.01f, 0.f, .0001f, .01255f, .025f, .05f };
    Data conversion(original, Domain::AttenuationMmInv);
    Pipeline forward;
    forward.append(makeMuToHuStage(reference, location));
    forward.run(conversion.volume, conversion.context);
    conversion.download();
    const double gain = (reference.water_hu - reference.air_hu) / (reference.water_mu_mm_inv - reference.air_mu_mm_inv);
    for (size_t i = 0; i < original.size(); ++i) {
        double expected = reference.air_hu + (double(original[i]) - reference.air_mu_mm_inv) * gain;
        require(std::abs(conversion.volume.host_data[i] - expected) < 2.e-4, "mu -> HU reference mismatch");
    }
    Pipeline backward;
    backward.append(makeHuToMuStage(reference, location));
    backward.run(conversion.volume, conversion.context);
    conversion.download();
    for (size_t i = 0; i < original.size(); ++i)
        require(std::abs(conversion.volume.host_data[i] - original[i]) < 1.e-8, "mu/HU round trip or negative value lost");
    Data changed({ .02f }, Domain::AttenuationMmInv);
    Pipeline alternate;
    alternate.append(makeMuToHuStage({ .04 }, location));
    alternate.run(changed.volume, changed.context);
    changed.download();
    require(std::abs(changed.volume.host_data[0] + 500) < 1.e-4, "water reference not configurable");
    // Same-chain calibration then correction; first four voxels are air, last four water.
    const std::vector<float> measured = { -900, -910, -890, -900, 90, 100, 110, 100 };
    Data sample(measured, Domain::HounsfieldUnits);
    sample.volume.grid.dimensions = { 4, 2, 1 };
    const CalibrationRoi air { { 0, 0, 0 }, { 4, 1, 1 } }, water { { 0, 1, 0 }, { 4, 2, 1 } };
    auto result = std::make_shared<HuCalibrationResult>();
    Pipeline calibration;
    calibration.append(makeAirWaterHuCalibrationStage(air, water, result, {}, location));
    calibration.run(sample.volume, sample.context);
    sample.download();
    require(std::equal(measured.begin(), measured.end(), sample.volume.host_data), "calibration modified voxel values");
    require(result->valid && result->air_voxels == 4 && result->water_voxels == 4 && std::abs(result->measured_air_hu + 900) < 1.e-12 && std::abs(result->measured_water_hu - 100) < 1.e-12, "calibration ROI means/counts wrong");
    require(std::abs(result->coefficients.scale - 1) < 1.e-12 && std::abs(result->coefficients.offset + 100) < 1.e-12, "air/water fitted coefficients wrong");
    Data chainData(measured, Domain::HounsfieldUnits);
    chainData.volume.grid.dimensions = { 4, 2, 1 };
    Pipeline chain;
    chain.append(makeAirWaterHuCalibrationStage(air, water, result, {}, location));
    chain.append(makeHuCorrectionStage(result, location));
    chain.run(chainData.volume, chainData.context);
#ifdef YK_INTENSITY_CUDA
    if (location == Location::Device)
        require(chainData.transfer.uploads == 1 && chainData.transfer.downloads == 0, "calibration/correction downloaded the volume");
#endif
    chainData.download();
    for (size_t i = 0; i < measured.size(); ++i)
        require(std::abs(chainData.volume.host_data[i] - (measured[i] - 100)) < 1.e-5, "same-chain HU correction mismatch");
    Data later({ -900, 100, 600 }, Domain::HounsfieldUnits);
    Pipeline reuse;
    reuse.append(makeHuCorrectionStage(result->coefficients, location));
    reuse.run(later.volume, later.context);
    later.download();
    require(later.volume.host_data[0] == -1000 && later.volume.host_data[1] == 0 && later.volume.host_data[2] == 500, "saved coefficients not reusable");
    // Upper intervals include their lower thresholds; named labels survive reset.
    Data classes({ -1001, -500, -499, 199, 200, 201, 4096 }, Domain::HounsfieldUnits);
    Pipeline segment;
    segment.append(makeHuThresholdStage({ { -500, 200 }, { { 7, "air" }, { 11, "soft" }, { 23, "bone" } } }, location));
    segment.run(classes.volume, classes.context);
    classes.download();
    const float expected[] = { 7, 11, 11, 11, 23, 23, 23 };
    require(classes.volume.domain == Domain::Labels, "segmentation kept HU domain");
    require(std::equal(std::begin(expected), std::end(expected), classes.volume.host_data), "HU threshold boundary/labels incorrect");
    require(classes.volume.label_mapping && classes.volume.label_mapping->classes[2].name == "bone" && classes.volume.label_mapping->thresholds_hu[1] == 200, "label metadata lost after reset");
    Pipeline invalidDomain;
    invalidDomain.append(makeHuToMuStage(reference, location));
    rejects([&] { invalidDomain.run(classes.volume, classes.context); });
    // Failure must invalidate old calibration; a failed attempt cannot be reused.
    Data broken({ -900, -910, -890, -900, -900, -900, -900, -900 }, Domain::HounsfieldUnits);
    broken.volume.grid.dimensions = { 4, 2, 1 };
    rejects([&] { calibration.run(broken.volume, broken.context); });
    require(!result->valid, "failed fit retained a valid stale calibration");
    Data afterFailure({ 0, 10 }, Domain::HounsfieldUnits);
    Pipeline invalidCorrection;
    invalidCorrection.append(makeHuCorrectionStage(result, location));
    rejects([&] { invalidCorrection.run(afterFailure.volume, afterFailure.context); });
    // Nonfinite/overflow check is atomic with respect to image mutation.
    Data nonfinite({ 2, 3, std::numeric_limits<float>::infinity() }, Domain::HounsfieldUnits);
    Pipeline correction;
    correction.append(makeHuCorrectionStage(HuLinearCoefficients { 2, 3 }, location));
    rejects([&] { correction.run(nonfinite.volume, nonfinite.context); });
    nonfinite.download();
    require(nonfinite.volume.host_data[0] == 2 && nonfinite.volume.host_data[1] == 3, "failed correction partially mutated values");
    Data overflowing({ 1, 2 }, Domain::HounsfieldUnits);
    Pipeline overflow;
    overflow.append(makeHuCorrectionStage(HuLinearCoefficients { std::numeric_limits<double>::max(), 0 }, location));
    rejects([&] { overflow.run(overflowing.volume, overflowing.context); });
    overflowing.download();
    require(overflowing.volume.host_data[0] == 1, "overflow partially mutated image");
    Data badRoi(measured, Domain::HounsfieldUnits);
    badRoi.volume.grid.dimensions = { 4, 2, 1 };
    Pipeline overlap;
    overlap.append(makeAirWaterHuCalibrationStage(air, air, result, {}, location));
    rejects([&] { overlap.run(badRoi.volume, badRoi.context); });
    // Nontrivial row/slice strides and multiple CUDA reduction blocks.
    std::vector<float> large(31 * 19 * 17);
    double airSum = 0, waterSum = 0;
    size_t airCount = 0, waterCount = 0;
    for (int z = 0; z < 17; ++z)
        for (int y = 0; y < 19; ++y)
            for (int x = 0; x < 31; ++x) {
                float value = float((x < 14 ? -970 : 45) + (x % 7) * .25 + (y % 5) * .125 + z * .5);
                large[(z * 19 + y) * 31 + x] = value;
                if (x >= 2 && x < 14 && y >= 1 && y < 18 && z >= 2 && z < 16) {
                    airSum += value;
                    ++airCount;
                }
                if (x >= 16 && x < 30 && y >= 2 && y < 17 && z >= 1 && z < 15) {
                    waterSum += value;
                    ++waterCount;
                }
            }
    Data statistics(std::move(large), Domain::HounsfieldUnits);
    statistics.volume.grid.dimensions = { 31, 19, 17 };
    Pipeline largeCalibration;
    largeCalibration.append(makeAirWaterHuCalibrationStage({ { 2, 1, 2 }, { 14, 18, 16 } }, { { 16, 2, 1 }, { 30, 17, 15 } }, result, {}, location));
    largeCalibration.run(statistics.volume, statistics.context);
    require(result->air_voxels == airCount && result->water_voxels == waterCount && std::abs(result->measured_air_hu - airSum / airCount) < 1.e-10 && std::abs(result->measured_water_hu - waterSum / waterCount) < 1.e-10, "multi-block ROI reduction or 3D indexing wrong");
    // End-to-end keeps attenuation as the final domain for the existing RAW exporter.
    Data full({ .002f, .02f }, Domain::AttenuationMmInv);
    Pipeline fullchain;
    fullchain.append(makeMuToHuStage({ .02 }, location));
    fullchain.append(makeHuCorrectionStage(HuLinearCoefficients { 1.1, 15 }, location));
    fullchain.append(makeHuToMuStage({ .02 }, location));
    fullchain.run(full.volume, full.context);
    full.download();
    require(full.volume.domain == Domain::AttenuationMmInv, "full HU correction chain did not return attenuation");
    for (size_t i = 0; i < 2; ++i) {
        double hu = 1000 * (double(i ? .02f : .002f) / .02 - 1), mu = .02 * ((1.1 * hu + 15) / 1000 + 1);
        require(std::abs(full.volume.host_data[i] - mu) < 1.e-8, "full HU correction chain value wrong");
    }
    // Thresholds without labels generate stable interval IDs, including the
    // immediately adjacent float values on either side of an exact threshold.
    Data boundary({ std::nextafter(200.f, -INFINITY), 200.f,
                      std::nextafter(200.f, INFINITY) },
        Domain::HounsfieldUnits);
    Pipeline automaticLabels;
    automaticLabels.append(makeHuThresholdStage({ { 200 }, {} }, location));
    automaticLabels.run(boundary.volume, boundary.context);
    boundary.download();
    require(boundary.volume.host_data[0] == 0 && boundary.volume.host_data[1] == 1 && boundary.volume.host_data[2] == 1, "adjacent threshold values classified incorrectly");
    Data nanLabels({ 3, std::numeric_limits<float>::quiet_NaN() }, Domain::HounsfieldUnits);
    rejects([&] { automaticLabels.run(nanLabels.volume, nanLabels.context); });
    nanLabels.download();
    require(nanLabels.volume.host_data[0] == 3 && nanLabels.volume.domain == Domain::HounsfieldUnits,
        "failed segmentation published partial labels");
    for (const auto roi : { CalibrationRoi { { -1, 0, 0 }, { 1, 1, 1 } },
             CalibrationRoi { { 0, 0, 0 }, { 0, 1, 1 } },
             CalibrationRoi { { 0, 0, 0 }, { 5, 1, 1 } } }) {
        Data invalidRoi(measured, Domain::HounsfieldUnits);
        invalidRoi.volume.grid.dimensions = { 4, 2, 1 };
        Pipeline p;
        p.append(makeAirWaterHuCalibrationStage(roi, water, result, {}, location));
        rejects([&] { p.run(invalidRoi.volume, invalidRoi.context); });
        require(!result->valid, "invalid ROI retained stale calibration");
    }
    std::cout << (location == Location::Host ? "CPU" : "CUDA") << " intensity stages passed\n";
}
int main()
{
    try {
        test(Location::Host);
#ifdef YK_INTENSITY_CUDA
        test(Location::Device);
        image::Options filter;
        rejects([&] { image::makeVolumeStage(filter, Domain::Labels); });
#endif
        rejects([] { makeMuToHuStage({ 0 }, Location::Host); });
        rejects([] { makeHuToMuStage({ .02, .02 }, Location::Host); });
        rejects([] { makeHuCorrectionStage(HuLinearCoefficients { -1, 0 }, Location::Host); });
        rejects([] { makeHuThresholdStage({ { 1, 1 }, {} }, Location::Host); });
        rejects([] { makeHuThresholdStage({ { 1 }, { { 1, "a" } } }, Location::Host); });
        rejects([] { makeHuThresholdStage({ {}, { { 16777217, "too large" } } }, Location::Host); });
        rejects([] { fitAirWaterHu(0, 0); });
        rejects([] { fitAirWaterHu(10, 0); });
        rejects([] { fitAirWaterHu(-900, -899.5); });
        require(fitAirWaterHu(-900, -899).valid, "minimum HU gap rejected");
        rejects([] { fitAirWaterHu(NAN, 0); });
        rejects([] { fitAirWaterHu(-1000, INFINITY); });
        rejects([] { fitAirWaterHu(-1000, 0, { 0, -1000, 1 }); });
        rejects([] { makeHuThresholdStage({ { NAN }, {} }, Location::Host); });
        rejects([] { makeHuThresholdStage({ { 1 }, { { 2, "a" }, { 2, "b" } } }, Location::Host); });
        auto nonstandard = fitAirWaterHu(-800, 200, { -950, 50, 1 });
        require(nonstandard.coefficients.scale == 1 && nonstandard.coefficients.offset == -150, "custom calibration targets ignored");
        for (const auto& p : intensityPluginDescriptions())
            pipeline::validatePluginInfo(p);
        std::cout << "Reference anchors, ROI calibration, reusable coefficients, labels, domain guards and failure validation passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
