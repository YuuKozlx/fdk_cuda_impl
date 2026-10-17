// Native thread-layout comparison; no reconstruction, plugins, or image export.
#include "primary/WarpSpectralTransport.cuh"
#include "config/SimConfig.hpp"
#include "XcomAttenuationProvider.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
using namespace yk::spectral;
using namespace yk::spectral::primary_detail;
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Buffer{
 T* p=nullptr; explicit Buffer(size_t n){ck(cudaMalloc(&p,n*sizeof(T)));}~Buffer(){cudaFree(p);}
 Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
 void upload(const std::vector<T>& v){ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T> read(size_t n){std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
struct Texture{
 cudaArray_t array=nullptr;cudaTextureObject_t id=0;
 Texture(const std::vector<unsigned char>& labels,const DeviceGeometry& g){
  auto channel=cudaCreateChannelDesc<unsigned char>();auto extent=make_cudaExtent(g.nx,g.ny,g.nz);
  ck(cudaMalloc3DArray(&array,&channel,extent));cudaMemcpy3DParms copy{};
  copy.srcPtr=make_cudaPitchedPtr(const_cast<unsigned char*>(labels.data()),g.nx,g.nx,g.ny);
  copy.dstArray=array;copy.extent=extent;copy.kind=cudaMemcpyHostToDevice;ck(cudaMemcpy3D(&copy));
  cudaResourceDesc resource{};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;
  cudaTextureDesc desc{};desc.filterMode=cudaFilterModePoint;desc.readMode=cudaReadModeElementType;
  for(auto& mode:desc.addressMode)mode=cudaAddressModeClamp;
  ck(cudaCreateTextureObject(&id,&resource,&desc,nullptr));
 }
 ~Texture(){cudaDestroyTextureObject(id);cudaFreeArray(array);}
};
struct Args{
 DeviceGeometry g;const unsigned char* labels;const int* map;const float *mu,*spectrum,*energies;
 int noise;unsigned seed,noiseSeed;float *expected,*sampled,*air;unsigned char* mask;unsigned long long* profile=nullptr;
};
using Launch=void(*)(const Args&);
struct Variant{const char* name;int group,block;Launch launch;cudaFuncAttributes resources{};std::vector<float> times;};
template<int M,int Group,int Block> void candidate(const Args& a){
 constexpr int Groups=Block/Group;constexpr bool SharedPaths=M!=10||Group!=32;constexpr size_t Shared=SharedPaths?M*Block*sizeof(float):0;
 if(a.profile){warpPrimaryKernel<M,128,Groups,(M==10&&Group==32),false,true,SharedPaths,Group><<<dim3((a.g.nu*a.g.nv+Groups-1)/Groups,a.g.views),Block,Shared>>>(a.g,a.labels,a.map,a.mu,a.spectrum,a.energies,a.noise,0,a.seed,a.noiseSeed,a.expected,a.sampled,a.air,a.mask,a.profile);ck(cudaGetLastError());return;}
 warpPrimaryKernel<M,128,Groups,(M==10&&Group==32),false,false,SharedPaths,Group><<<dim3((a.g.nu*a.g.nv+Groups-1)/Groups,a.g.views),Block,Shared>>>(
 a.g,a.labels,a.map,a.mu,a.spectrum,a.energies,a.noise,0,a.seed,a.noiseSeed,a.expected,a.sampled,a.air,a.mask);
 ck(cudaGetLastError());
}
template<int M,int Group,int Block> void add(std::vector<Variant>& variants,const char* name){
 constexpr bool SharedPaths=M!=10||Group!=32;
 auto kernel=warpPrimaryKernel<M,128,Block/Group,(M==10&&Group==32),false,false,SharedPaths,Group>;
 constexpr int bytes=SharedPaths?M*Block*sizeof(float):0;
 if(bytes>48*1024)ck(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
 Variant v{name,Group,Block,candidate<M,Group,Block>};ck(cudaFuncGetAttributes(&v.resources,kernel));variants.push_back(v);
}
template<int M> void baseline(const Args& a){
 // Match production's 10-material coefficient-cache specialization.
 launchWarpPrimarySized<M,128,M==10>(a.g,a.labels,a.map,a.mu,a.spectrum,a.energies,a.noise,0,a.seed,a.noiseSeed,a.expected,a.sampled,a.air,a.mask,a.g.views,false);
 ck(cudaGetLastError());
}
template<int M> std::vector<Variant> variants(){
 std::vector<Variant> v{{"baseline",32,128,baseline<M>}};
 add<M,32,128>(v,"g32_b128");
 add<M,4,64>(v,"g4_b64");add<M,4,128>(v,"g4_b128");add<M,4,256>(v,"g4_b256");
 add<M,8,64>(v,"g8_b64");add<M,8,128>(v,"g8_b128");add<M,8,256>(v,"g8_b256");
 add<M,16,64>(v,"g16_b64");add<M,16,128>(v,"g16_b128");add<M,16,256>(v,"g16_b256");
 ck(cudaFuncGetAttributes(&v[0].resources,warpPrimaryKernel<M,128,4,M==10>));return v;
}
struct Events{
 cudaEvent_t start{},end{};Events(){ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));}~Events(){cudaEventDestroy(start);cudaEventDestroy(end);}
 float time(Launch launch,const Args& a){ck(cudaEventRecord(start));launch(a);ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,start,end));return ms;}
};
int main(int argc,char** argv){try{
 if(argc<5||argc>6)throw std::runtime_error("Usage: multispectrum_benchmark_warp <RAW-label.toml> <rows> <rays> <results.csv> [all|policy|controlled:N]");
 const bool controlled=argc==6&&std::string(argv[5]).rfind("controlled:",0)==0;
 auto c=loadConfig(argv[1],false);auto& p=c.geometry.parameters;
 if(controlled){
  const int count=std::stoi(std::string(argv[5]).substr(11));
  const auto materials=c.projection.materials;
  if(count<1||count>int(materials.size())||count>64)throw std::runtime_error("Controlled count exceeds available distinct materials");
  c.projection.materials.assign(materials.begin(),materials.begin()+count);
  for(int i=0;i<count;++i)c.projection.materials[i].label=static_cast<unsigned char>(i);
  c.projection.quantum_noise.enabled=false;c.projection.focal_spot.enabled=false;c.projection.focal_spot.defocus_enabled=false;
 }
 if(c.geometry.kind!=GeometryKind::FlatCbct&&c.geometry.kind!=GeometryKind::FlatHelical)throw std::runtime_error("Only flat circular/helical geometry supported by this benchmark");
 if(c.projection.detector_response.enabled)throw std::runtime_error("Disable detector response for this isolated primary benchmark");
 const int rows=std::stoi(argv[2]),rays=std::stoi(argv[3]);if(rows<=0||rays<=0)throw std::runtime_error("Positive rows/rays required");
 p.pixel_u_mm*=double(p.detector_u)/1024;p.detector_u=1024;
 p.pixel_v_mm*=double(p.detector_v)/rows;p.detector_v=rows;p.views=12;p.views_per_turn=12;
 c.projection.sampling.samples_per_pixel=rays;
 const int nm=int(c.projection.materials.size());if(nm<1||nm>64)throw std::runtime_error("Benchmark requires 1..64 materials");
 auto spectrum=loadProjectionSpectrum(c.projection);const int ne=int(spectrum.size());if(ne<=0||ne>128)throw std::runtime_error("Comparison requires <=128 energy bins");
 auto g=makeDeviceGeometry(c,nm,ne);
 const size_t voxels=size_t(g.nx)*g.ny*g.nz,n=size_t(g.nu)*g.nv*g.views;
 if(n*13>256ULL*1024*1024)throw std::runtime_error("Output buffers exceed 256 MiB benchmark budget");
 std::vector<unsigned char> labels(voxels);
 if(controlled){
  // Identical physical envelope and grid for every count; concentric material
  // shells exercise many path channels without changing voxel traversal length.
  if(g.nx!=512||g.ny!=512||g.nz!=400||g.vx!=.5f||g.vy!=.5f||g.vz!=.5f)throw std::runtime_error("Controlled benchmark requires 512x512x400 / 0.5mm grid");
  for(int z=0;z<g.nz;++z)for(int y=0;y<g.ny;++y)for(int x=0;x<g.nx;++x){
   const float xx=(x+.5f-.5f*g.nx)*g.vx,yy=(y+.5f-.5f*g.ny)*g.vy,zz=(z+.5f-.5f*g.nz)*g.vz;
   const float radius=std::sqrt(xx*xx+yy*yy);
   labels[(size_t(z)*g.ny+y)*g.nx+x]=(radius<96.f&&std::abs(zz)<80.f)?static_cast<unsigned char>(std::min(nm-1,int(radius*nm/96.f))):255;
  }
 }else{
  if(std::filesystem::file_size(c.projection.label_volume)!=voxels)throw std::runtime_error("Expected uint8 RAW labels");
  std::ifstream input(c.projection.label_volume,std::ios::binary);
  if(!input.read(reinterpret_cast<char*>(labels.data()),labels.size()))throw std::runtime_error("Cannot read labels");
 }
 std::vector<int> map(256,-1);std::vector<float> mu(nm*ne),sw(ne),en(ne);std::vector<double> energies;
 for(const auto& point:spectrum)energies.push_back(point.energy_keV);
 XcomAttenuationProvider provider({});std::string error;
 for(int m=0;m<nm;++m){const auto& material=c.projection.materials[m];map[material.label]=m;
  std::vector<double> values;if(!provider.query(material,energies,values,error))throw std::runtime_error(error);
  for(int e=0;e<ne;++e)mu[m*ne+e]=float(values[e]*material.density_g_cm3);
 }
 for(auto label:labels)if(map[label]<0&&!(controlled&&label==255))throw std::runtime_error("Unknown volume label");
 for(int e=0;e<ne;++e){const auto& point=spectrum[e];en[e]=float(point.energy_keV);sw[e]=float(point.fluence_per_keV_cm2_mAs_at_1m*point.bin_width_keV*c.projection.mAs_per_view*p.pixel_u_mm*p.pixel_v_mm/100.);}
 Buffer<unsigned char> dl(labels.size()),mask(n);dl.upload(labels);Texture tex(labels,g);g.label_texture=tex.id;g.use_label_texture=1;
 Buffer<int> dm(256);dm.upload(map);Buffer<float> dmu(mu.size()),ds(ne),de(ne),expected(n),sampled(n),air(n);dmu.upload(mu);ds.upload(sw);de.upload(en);
 Args a{g,dl.p,dm.p,dmu.p,ds.p,de.p,c.projection.quantum_noise.enabled?1:0,c.projection.sampling.seed,c.projection.quantum_noise.seed,expected.p,sampled.p,air.p,mask.p};
 auto v=nm<=10?variants<10>():nm<=32?variants<32>():variants<64>();
 if(controlled){
  const auto layout=selectPrimaryLaunchLayout(nm,ne,rays);
  v.erase(std::remove_if(v.begin(),v.end(),[&](const Variant& x){return std::string(x.name)!="baseline"&&(x.group!=layout.threads_per_pixel||x.block!=layout.threads_per_block);}),v.end());
 }
 else if(argc==6&&std::string(argv[5])=="policy")v.erase(std::remove_if(v.begin(),v.end(),[&](const Variant& x){return x.group!=32&&x.block!=(nm>32&&x.group==16?64:128);}),v.end());
 else if(argc==6&&std::string(argv[5])!="all")throw std::runtime_error("Invalid variant selection");
 Events timer;v[0].launch(a);ck(cudaDeviceSynchronize());
 const auto ref=expected.read(n),refNoise=sampled.read(n),refAir=air.read(n);const auto refMask=mask.read(n);
 std::cout<<std::setprecision(9)<<"materials="<<nm<<" bins="<<ne<<" detector="<<g.nu<<"x"<<g.nv<<" views="<<g.views<<" rays="<<g.samples<<" noise="<<a.noise<<" focal_epsilon="<<g.focal_epsilon<<std::endl;
 for(auto& variant:v){
  const float warmup=timer.time(variant.launch,a);double maxRel=0,maxLog=0;size_t invalid=0;
  const auto x=expected.read(n),y=sampled.read(n),z=air.read(n);const auto h=mask.read(n);
  for(size_t i=0;i<n;++i){
   const std::array<std::pair<float,float>,3> pairs{{{x[i],ref[i]},{y[i],refNoise[i]},{z[i],refAir[i]}}};
   for(const auto& pair:pairs){
    if(!std::isfinite(pair.first)||!std::isfinite(pair.second))++invalid;
    maxRel=std::max(maxRel,std::abs(double(pair.first)-pair.second)/std::max(1e-20,std::abs(double(pair.second))));
   }
   const double actualLog=-std::log(std::max(1e-30,double(y[i])/std::max(1e-30,double(z[i]))));
   const double refLog=-std::log(std::max(1e-30,double(refNoise[i])/std::max(1e-30,double(refAir[i]))));
   maxLog=std::max(maxLog,std::abs(actualLog-refLog));if(h[i]!=refMask[i])++invalid;
  }
  std::cout<<"validate "<<variant.name<<" warmup_ms="<<warmup<<" max_relative="<<maxRel<<" max_log_absolute="<<maxLog<<" invalid="<<invalid<<" regs="<<variant.resources.numRegs<<" local="<<variant.resources.localSizeBytes<<" static_shared="<<variant.resources.sharedSizeBytes<<std::endl;
  if(invalid||maxRel>3e-5||maxLog>3e-5)throw std::runtime_error("Candidate numerical validation failed");
 }
 // Rotating and reversing order reduces monotonic thermal/frequency bias.
 for(int round=0;round<3;++round){for(size_t j=0;j<v.size();++j){
  const size_t index=round%2?(v.size()-1-j+round*3)%v.size():(j+round*3)%v.size();
  auto& variant=v[index];const auto ms=timer.time(variant.launch,a);variant.times.push_back(ms);
  std::cout<<"timing round="<<round<<" variant="<<variant.name<<" ms="<<ms<<std::endl;
 }}
 if(controlled){
  Buffer<unsigned long long> counters(WarpProfileCounterCount);ck(cudaMemset(counters.p,0,WarpProfileCounterCount*sizeof(unsigned long long)));
  a.profile=counters.p;v.back().launch(a);ck(cudaDeviceSynchronize());const auto values=counters.read(WarpProfileCounterCount);a.profile=nullptr;
  unsigned long long sum=0;for(int i=0;i<6;++i)sum+=values[i];
  std::cout<<"profile_pixel_groups="<<values[6]<<" rays="<<values[7]<<" hit_rays="<<values[8]<<" visited_voxels="<<values[9]<<std::endl;
  for(int i=0;i<6;++i)std::cout<<"profile_phase="<<i<<" cycles="<<values[i]<<" percent="<<(sum?100.*values[i]/sum:0)<<std::endl;
 }
 const bool header=!std::filesystem::exists(argv[4]);std::ofstream csv(argv[4],std::ios::app);if(!csv)throw std::runtime_error("Cannot open CSV");
 if(header)csv<<"materials,rows,rays,variant,group,block,registers,local_bytes,static_shared,dynamic_shared,run1_ms,run2_ms,run3_ms,median_ms\n";
 for(auto& variant:v){auto sorted=variant.times;std::sort(sorted.begin(),sorted.end());
  csv<<nm<<','<<rows<<','<<rays<<','<<variant.name<<','<<variant.group<<','<<variant.block<<','<<variant.resources.numRegs<<','<<variant.resources.localSizeBytes<<','<<variant.resources.sharedSizeBytes<<','<<((std::string(variant.name)=="baseline"||(nm<=10&&variant.group==32))?0:(nm<=10?10:nm<=32?32:64)*variant.block*4)<<','<<variant.times[0]<<','<<variant.times[1]<<','<<variant.times[2]<<','<<sorted[1]<<'\n';
  std::cout<<"result "<<variant.name<<" median_ms="<<sorted[1]<<std::endl;
 }
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
