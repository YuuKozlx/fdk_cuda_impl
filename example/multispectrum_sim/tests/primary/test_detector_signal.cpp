#include "TestChecks.hpp"
#include "primary/DetectorSignal.hpp"
#include "primary/ProjectionOutput.hpp"
#include "primary/AirReferenceReader.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
using namespace yk::spectral;
using namespace yk::spectral::primary_detail;
using namespace primary_test;
namespace {
void crosstalk() {
    CrosstalkConfig c; c.enabled = true;
    c.kernel_u = {0.25,0.5,0.25}; c.kernel_v = {1.0};
    std::vector<float> constant(15, 7.f);
    applyCrosstalkSignal(constant, 5, 3, c);
    for (float x : constant) near(x,7,1e-6,"DC preservation, including border");
    std::vector<float> impulse = {0,0,1,0,0};
    applyCrosstalkSignal(impulse,5,1,c);
    const double expected[] = {0,.25,.5,.25,0};
    for (int i=0;i<5;++i) near(impulse[i],expected[i],1e-6,"impulse response");
    std::vector<float> border = {1,0,0};
    applyCrosstalkSignal(border,3,1,c);
    near(border[0],2.0/3.0,1e-6,"border renormalization");
    near(border[1],.25,1e-6,"border neighbor");
    c.kernel_v = {0.25,0.5,0.25}; c.kernel_u = {1.0};
    std::vector<float> vertical = {0,0,1,0,0};
    applyCrosstalkSignal(vertical,1,5,c);
    for (int i=0;i<5;++i) near(vertical[i],expected[i],1e-6,"V-axis convolution");
    c.enabled = false;
    const auto before = vertical; applyCrosstalkSignal(vertical,1,5,c);
    require(vertical == before,"disabled crosstalk");
}
void afterglow() {
    DetectorPostprocessConfig::AfterglowConfig c;
    c.enabled = true; c.p = .2; c.exposure_time_ms = 1; c.frame_time_ms = 1;
    c.weights = {1}; c.time_constants_ms = {2};
    std::vector<std::vector<float>> state;
    std::vector<float> input = {10};
    applyAfterglowSignal(input,c,state);
    // Continuous trap ODE: dq/dt = p*S/T - q/tau, integrated over T=1.
    const double trapped = 4*(1-std::exp(-.5));
    near(input[0],10-trapped,2e-6,"first-frame charge balance");
    near(state[0][0],trapped,2e-6,"remaining trapped signal");
    double emitted = input[0];
    double previous = 100;
    for (int i=0;i<40;++i) {
        input = {0}; applyAfterglowSignal(input,c,state);
        require(input[0]>=0 && input[0]<=previous,"monotone afterglow decay");
        emitted+=input[0]; previous=input[0];
    }
    near(emitted+state[0][0],10,5e-6,"charge conserved over frames");
    c.initialization = "steady_state"; state.clear(); input={10};
    applyAfterglowSignal(input,c,state);
    near(input[0],10,3e-6,"steady-state constant exposure");
    c.initialization="zero"; state.clear(); c.frame_time_ms=3; input={10};
    applyAfterglowSignal(input,c,state);
    near(state[0][0],trapped*std::exp(-1),2e-6,"readout-period decay");
    c.enabled=false; auto oldState=state; auto old=input;
    applyAfterglowSignal(input,c,state);
    require(input==old && state==oldState,"disabled afterglow does not advance state");
}
void das() {
    DetectorPostprocessConfig::DasConfig c;
    c.enabled=true; c.gain_electrons_per_keV=10; c.offset_electrons=5;
    c.saturation_electrons=100; c.adc_lsb_electrons=4;
    c.electronic_noise_std_electrons=0;
    std::mt19937 random(12); std::vector<float> signal={1,20};
    applyDasSignal(signal,c,random);
    near(signal[0],1.6,1e-6,"gain, offset and ADC quantization");
    near(signal[1],10,1e-6,"saturation");
    c.offset_electrons=0; c.saturation_electrons=0; c.adc_lsb_electrons=0;
    c.electronic_noise_std_electrons=20;
    std::vector<float> a(65536,100), b=a;
    std::mt19937 ra(123),rb(123); applyDasSignal(a,c,ra);applyDasSignal(b,c,rb);
    require(a==b,"DAS fixed-seed reproducibility");
    double mean=0,var=0;
    for(float x:a)mean+=x;mean/=a.size();
    for(float x:a)var+=(x-mean)*(x-mean);var/=a.size();
    near(mean,100,.04,"DAS noise mean"); near(var,4,.12,"DAS noise variance");
    c.enabled=false; auto before=a;auto engine=ra;
    applyDasSignal(a,c,ra); require(a==before && ra==engine,"disabled DAS");
}
void thickness(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    DetectorLayerConfig c; c.thickness_mm=.7;
    auto uniform=loadLayerThickness(c,3);
    for(float x:uniform) near(x,.7,1e-6,"uniform layer thickness");
    c.thickness_mm=-.1;
    rejects([&]{loadLayerThickness(c,3);},"negative uniform thickness");
    c.thickness_mm=std::numeric_limits<double>::quiet_NaN();
    rejects([&]{loadLayerThickness(c,3);},"NaN uniform thickness");
    c.thickness_mm=.7;
    rejects([&]{loadLayerThickness(c,0);},"zero thickness-map dimensions");
    c.thickness_map_file=directory/"thickness.raw";
    auto save=[&](std::vector<float> v) {
        std::ofstream out(c.thickness_map_file,std::ios::binary|std::ios::trunc);
        out.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(float));
    };
    save({.1f,.2f,.3f}); require(loadLayerThickness(c,3)==std::vector<float>({.1f,.2f,.3f}),"thickness map");
    save({.1f,.2f}); rejects([&]{loadLayerThickness(c,3);},"short thickness map");
    save({.1f,.2f,.3f,.4f}); rejects([&]{loadLayerThickness(c,3);},"long thickness map");
    save({.1f,-1,.3f}); rejects([&]{loadLayerThickness(c,3);},"negative thickness");
    save({.1f,std::numeric_limits<float>::quiet_NaN(),.3f});
    rejects([&]{loadLayerThickness(c,3);},"NaN thickness");
}
void output(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    SimulationConfig c; auto& g=c.geometry.parameters;
    g.detector_u=2;g.detector_v=1;g.views=2;
    c.projection.save_metadata=true;c.projection.energy_output_file=directory/"energy.raw";
    c.projection.air_output_file=directory/"air.raw";
    EnergyProjectionOutput sink; const auto path=directory/"attenuation.raw";
    sink.prepare({c,path});
    ProjectionFrame f;f.width=2;f.height=1;f.view=0;
    f.signal_energy={50,25};f.reference_energy={100,100};
    sink.write(f,{c,path});
    f.view=1;f.signal_energy={100,50};f.reference_energy={200,200};
    sink.write(f,{c,path});sink.finish();
    std::ifstream in(path,std::ios::binary);float values[2];in.read(reinterpret_cast<char*>(values),sizeof(values));
    require(bool(in),"attenuation output read");
    near(values[0],std::log(2.0),1e-6,"negative log pixel 0");
    near(values[1],std::log(4.0),1e-6,"negative log pixel 1");
    require(std::filesystem::file_size(c.projection.air_output_file)==4*sizeof(float),"per-view air reference layout");
    std::ifstream air(c.projection.air_output_file,std::ios::binary);
    float reference[4]{};air.read(reinterpret_cast<char*>(reference),sizeof(reference));
    require(bool(air) && reference[2]==200 && reference[3]==200,"second-view air values");
    std::ifstream metadata(c.projection.air_output_file.string()+".json");
    const std::string text((std::istreambuf_iterator<char>(metadata)),{});
    require(text.find("\"frames\": 2")!=std::string::npos,"air metadata frame count");
    sink.reset();sink.prepare({c,path});f.domain=ProjectionSignalDomain::PhotonCounts;
    rejects([&]{sink.write(f,{c,path});},"count signal to energy sink");
    rejects([&]{sink.finish();},"incomplete output");
    sink.reset();
}
void airInput(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const auto path=directory/"air.raw";
    auto save=[&](std::vector<float> values) {
        std::ofstream file(path,std::ios::binary|std::ios::trunc);
        file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
    };
    std::vector<float> air;
    save({10,20});
    { AirReferenceReader reader(path,2,3);
      for(int i=0;i<3;++i){reader.read(i,air);require(air==std::vector<float>({10,20}),"legacy air broadcast");}
    }
    save({10,20,30,40,50,60});
    { AirReferenceReader reader(path,2,3);
      reader.read(1,air);require(air==std::vector<float>({30,40}),"per-view air selection");
      reader.read(2,air);require(air==std::vector<float>({50,60}),"last-view air selection");
      rejects([&]{reader.read(3,air);},"out-of-range air view");
    }
    save({1,2,3});rejects([&]{AirReferenceReader reader(path,2,3);},"incomplete air file");
    save({10,0});{AirReferenceReader reader(path,2,3);rejects([&]{reader.read(0,air);},"nonpositive air");}
}
}
int main(int argc,char** argv) {
    try {
        require(argc>=2,"module selection required");
        const std::string test=argv[1];
        if(test=="crosstalk")crosstalk();
        else if(test=="afterglow")afterglow();
        else if(test=="das")das();
        else if(test=="thickness" && argc==3)thickness(argv[2]);
        else if(test=="output" && argc==3)output(argv[2]);
        else if(test=="air_input" && argc==3)airInput(argv[2]);
        else throw std::runtime_error("unknown detector signal test");
        std::cout<<test<<": PASS\n";return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
