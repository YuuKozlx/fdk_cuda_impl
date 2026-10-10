#include "TestChecks.hpp"
#include "primary/SpectralTransport.cuh"
#include "primary/DetectorResponse.cuh"
#include <array>
#include <iostream>
#include <vector>
using namespace yk::spectral::primary_detail;
using namespace primary_test;
namespace {
void ck(cudaError_t error) {
    if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> class Device {
public:
    explicit Device(size_t n):size_(n){ck(cudaMalloc(&pointer_,n*sizeof(T)));}
    explicit Device(const std::vector<T>& data):Device(data.size()){upload(data);}
    ~Device(){cudaFree(pointer_);}
    Device(const Device&)=delete;
    T* get()const{return pointer_;}
    void upload(const std::vector<T>& data){require(data.size()==size_,"device upload size");ck(cudaMemcpy(pointer_,data.data(),size_*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> read()const{std::vector<T> out(size_);ck(cudaMemcpy(out.data(),pointer_,size_*sizeof(T),cudaMemcpyDeviceToHost));return out;}
private:T* pointer_=nullptr;size_t size_;
};
void sync(){ck(cudaGetLastError());ck(cudaDeviceSynchronize());}
__global__ void geometryProbe(DeviceGeometry g,float* out){
    const float3 rotated=rotateAxis(make_float3(1,0,0),make_float3(0,0,1),1.57079632679f);
    out[0]=rotated.x;out[1]=rotated.y;out[2]=rotated.z;
    const float3 point=phantomInverse(g,make_float3(-3,2,4),false);
    const float3 direction=phantomInverse(g,make_float3(-1,0,0),true);
    out[3]=point.x;out[4]=point.y;out[5]=point.z;
    out[6]=direction.x;out[7]=direction.y;out[8]=direction.z;
    g.tilt_u=.2f;g.tilt_v=.3f;g.tilt_n=.1f;
    float3 u,v,n;detectorFrame(g,.4f,u,v,n);
    out[9]=u.x*u.x+u.y*u.y+u.z*u.z;
    out[10]=v.x*v.x+v.y*v.y+v.z*v.z;
    out[11]=n.x*n.x+n.y*n.y+n.z*n.z;
    out[12]=u.x*v.x+u.y*v.y+u.z*v.z;
    out[13]=u.x*n.x+u.y*n.y+u.z*n.z;
    out[14]=v.x*n.x+v.y*n.y+v.z*n.z;
    const float3 cross=cross3(u,v);out[15]=cross.x*n.x+cross.y*n.y+cross.z*n.z;
}
void geometry(){
    DeviceGeometry g{};g.phantom_x=1;g.phantom_y=2;g.phantom_z=3;g.phantom_rz=1.57079632679f;
    Device<float> result(16);geometryProbe<<<1,1>>>(g,result.get());sync();auto x=result.read();
    const double expected[]={0,1,0,1,1,1,0,1,0,1,1,1,0,0,0,1};
    for(int i=0;i<16;++i)near(x[i],expected[i],2e-6,"geometry component "+std::to_string(i));
}
struct Ray {float3 source,direction;};
__global__ void pathProbe(DeviceGeometry g,const unsigned char* labels,const int* map,
    const Ray* rays,int count,float* out){
    const int i=threadIdx.x;if(i>=count)return;
    float paths[2]={};
    const float3 lo=make_float3(-2,-3,-4),hi=make_float3(2,3,4);
    const bool hit=accumulatePaths<2>(g,labels,map,rays[i].source,rays[i].direction,lo,hi,paths);
    out[i*3]=paths[0];out[i*3+1]=paths[1];out[i*3+2]=hit?1.f:0.f;
}
void paths(){
    DeviceGeometry g{};g.nx=g.ny=g.nz=2;g.vx=2;g.vy=3;g.vz=4;
    std::vector<unsigned char> labels={1,2,1,2,1,2,1,2};
    std::vector<int> map(256,-1);map[1]=0;map[2]=1;
    const float norm=std::sqrt(29.f);
    std::vector<Ray> rays={
        {make_float3(-10,0,0),make_float3(1,0,0)},
        {make_float3(10,0,0),make_float3(-1,0,0)},
        {make_float3(-10,4,0),make_float3(1,0,0)},
        {make_float3(.5f,0,0),make_float3(1,0,0)},
        {make_float3(-2,-3,-4),make_float3(2/norm,3/norm,4/norm)}};
    Device<unsigned char> dl(labels);Device<int> dm(map);Device<Ray> dr(rays);Device<float> result(rays.size()*3);
    pathProbe<<<1,32>>>(g,dl.get(),dm.get(),dr.get(),rays.size(),result.get());sync();auto linear=result.read();
    const double expected[]={.2,.2,1,.2,.2,1,0,0,0,0,.15,1,norm*.1,norm*.1,1};
    for(size_t i=0;i<linear.size();++i)near(linear[i],expected[i],2e-6,"anisotropic path "+std::to_string(i));
    // Exercise the production point-texture lookup against the same analytic paths.
    cudaArray_t array=nullptr;cudaTextureObject_t texture=0;
    try{
        const auto desc=cudaCreateChannelDesc<unsigned char>();auto extent=make_cudaExtent(2,2,2);
        ck(cudaMalloc3DArray(&array,&desc,extent));cudaMemcpy3DParms copy{};
        copy.srcPtr=make_cudaPitchedPtr(labels.data(),2,2,2);copy.dstArray=array;copy.extent=extent;copy.kind=cudaMemcpyHostToDevice;ck(cudaMemcpy3D(&copy));
        cudaResourceDesc resource{};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;
        cudaTextureDesc td{};td.readMode=cudaReadModeElementType;td.filterMode=cudaFilterModePoint;
        ck(cudaCreateTextureObject(&texture,&resource,&td,nullptr));g.use_label_texture=1;g.label_texture=texture;
        pathProbe<<<1,32>>>(g,dl.get(),dm.get(),dr.get(),rays.size(),result.get());sync();
        const auto textured=result.read();require(textured==linear,"texture and linear label traversal differ");
        cudaDestroyTextureObject(texture);cudaFreeArray(array);
    }catch(...){if(texture)cudaDestroyTextureObject(texture);if(array)cudaFreeArray(array);throw;}
}
__global__ void sampleProbe(float lambda,int count,float* counts,float* uniforms,float* normals){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const unsigned seed=12345U+static_cast<unsigned>(i)*747796405U;
    counts[i]=poissonSample(lambda,seed);uniforms[i]=uniform(seed);normals[i]=normal(seed);
}
void noise(){
    constexpr int n=65536;Device<float> counts(n),uniforms(n),normals(n);
    for(float lambda:{0.f,4.f,32.f,100.f}){
        sampleProbe<<<n/256,256>>>(lambda,n,counts.get(),uniforms.get(),normals.get());sync();
        const auto data=counts.read();double mean=0,var=0;
        for(float x:data){require(x>=0 && std::floor(x)==x,"noninteger or negative Poisson counts");mean+=x;}mean/=n;
        for(float x:data)var+=(x-mean)*(x-mean);var/=n;
        near(mean,lambda,std::max(.04,lambda*.015),"Poisson mean");
        near(var,lambda,std::max(.08,lambda*.05),"Poisson variance");
        sampleProbe<<<n/256,256>>>(lambda,n,counts.get(),uniforms.get(),normals.get());sync();
        require(counts.read()==data,"fixed-seed photon noise");
    }
    double mean=0,var=0;auto u=uniforms.read();
    for(float x:u){require(x>=0 && x<1,"uniform bounds");mean+=x;}mean/=n;
    for(float x:u)var+=(x-mean)*(x-mean);var/=n;
    near(mean,.5,.005,"uniform mean");near(var,1.0/12,.003,"uniform variance");
    auto z=normals.read();mean=var=0;
    for(float x:z)mean+=x;mean/=n;for(float x:z)var+=(x-mean)*(x-mean);var/=n;
    near(mean,0,.02,"normal mean");near(var,1,.04,"normal variance");
    Device<float> energy(std::vector<float>{0,400,800});
    applyQuantumNoiseAfterScatter<<<1,32>>>(energy.get(),3,40,123,0);sync();
    for(float x:energy.read())require(x>=0 && std::floor(x/40)==x/40,"scatter energy quantization");
    energy.upload({0,400,800});applyQuantumNoiseAfterScatter<<<1,32>>>(energy.get(),3,0,123,0);sync();
    require(energy.read()==std::vector<float>({0,400,800}),"zero-energy noise guard");
}
void response(){
    DeviceGeometry g{};g.nu=3;g.nv=1;g.ne=2;g.pu=100;g.pv=1;g.sid=200;g.sdd=300;
    Device<float> protective(std::vector<float>(3,.5f)),impurity(std::vector<float>(3,.2f)),sensor(std::vector<float>(3,.8f));
    Device<float> mp(std::vector<float>{.2f,.4f}),mi(std::vector<float>{.1f,.3f}),ms(std::vector<float>{1,2}),result(6);
    auto run=[&](bool oblique,bool includeImpurity){
        detectorEfficiencyKernel<<<1,32>>>(g,protective.get(),impurity.get(),sensor.get(),mp.get(),mi.get(),ms.get(),2,3,4,includeImpurity,oblique,result.get());
        sync();return result.read();
    };
    for(bool oblique:{false,true})for(bool include:{false,true}){
        const auto x=run(oblique,include);
        for(int pixel=0;pixel<3;++pixel)for(int e=0;e<2;++e){
            const double u=(pixel-1)*100;
            const double factor=oblique?std::sqrt(300*300+u*u)/300:1;
            const double entrance=.1*factor*(.5*2*(e?.4:.2)+(include?.2*3*(e?.3:.1):0));
            const double absorption=.1*factor*.8*4*(e?2:1);
            near(x[pixel*2+e],std::exp(-entrance)*(-std::expm1(-absorption)),2e-6,"layered detector efficiency");
        }
    }
    const auto initial=run(true,true);g.start_z=17;
    const auto shifted=run(true,true);
    for(int i=0;i<6;++i)near(shifted[i],initial[i],2e-6,"common axial translation invariance");
    g.start_z=0;g.start=.7f;
    const auto rotated=run(true,true);
    for(int i=0;i<6;++i)near(rotated[i],initial[i],2e-6,"common gantry rotation invariance");
    g.start=0;sensor.upload({0,0,0});
    for(float x:run(true,true))near(x,0,0,"zero scintillator thickness");
}
__global__ void focalProbe(float* result) {
    DeviceGeometry g{};g.tilt_u=.2f;g.tilt_v=-.1f;g.tilt_n=.3f;
    float3 u,v,n;detectorFrame(g,0.f,u,v,n);
    const float3 source=make_float3(0,-200,5);
    const auto a=sampleFlatFocalSource(source,u,v,2,3);
    const auto b=sampleFlatFocalSource(source,scale3(u,.417f),scale3(v,.8f),2,3);
    const auto delta=add3(a,scale3(source,-1));
    result[0]=delta.x*u.x+delta.y*u.y+delta.z*u.z;
    result[1]=delta.x*v.x+delta.y*v.y+delta.z*v.z;
    result[2]=delta.x*n.x+delta.y*n.y+delta.z*n.z;
    result[3]=a.x-b.x;result[4]=a.y-b.y;result[5]=a.z-b.z;
}
void focal(){
    Device<float> result(6);focalProbe<<<1,1>>>(result.get());sync();const auto delta=result.read();
    near(delta[0],2,3e-5,"focal displacement along tilted U");
    near(delta[1],3,3e-5,"focal displacement along tilted V");
    near(delta[2],0,3e-5,"no focal displacement along N");
    for(int i=3;i<6;++i)near(delta[i],0,3e-5,"pixel size independent focal dimensions");

    DeviceGeometry g{};g.nu=g.nv=1;g.nx=g.nz=1;g.ny=6;g.vx=g.vy=g.vz=1;
    g.nm=2;g.ne=2;g.samples=32;g.views_per_turn=1;g.rotation_direction=1;
    g.sid=10;g.sdd=20;g.pu=g.pv=.1f;g.tilt_u=.2f;
    g.normal_fu=1;g.normal_fv=.5f;g.defocus_fu=2;g.defocus_fv=.8f;g.focal_epsilon=.2f;
    g.geometry_flux=1;
    Device<unsigned char> labels(std::vector<unsigned char>{1,1,1,2,2,2}),mask(1);
    std::vector<int> map(256,-1);map[1]=0;map[2]=1;
    Device<int> dm(map);Device<float> mu(std::vector<float>{1,2,3,4}),spectrum(std::vector<float>{100,50}),energies(std::vector<float>{40,80});
    Device<float> expected(1),sampled(1),air(1);
    auto run=[&]{
        launchPrimaryKernel<2>(g,labels.get(),dm.get(),mu.get(),spectrum.get(),energies.get(),8000,0,0,123,456,expected.get(),sampled.get(),air.get(),mask.get());sync();
        return std::array<float,2>{expected.read()[0],air.read()[0]};
    };
    const auto flat=run();
    Device<DeviceConeView> view(std::vector<DeviceConeView>{{make_float3(0,-10,0),make_float3(0,10,0),
        make_float3(.1f,0,0),make_float3(0,-.1f*std::sin(.2f),.1f*std::cos(.2f))}});
    g.use_cone_views=1;g.cone_views=view.get();const auto cone=run();
    near(cone[0],flat[0],flat[0]*3e-6,"flat/ConeVec finite focal attenuation");
    near(cone[1],flat[1],flat[1]*3e-6,"flat/ConeVec finite focal air");
}
void transmission(){
    DeviceGeometry g{};g.nu=g.nv=1;g.nx=g.nz=1;g.ny=6;g.vx=g.vy=g.vz=1;
    g.nm=g.ne=2;g.samples=1;g.views_per_turn=1;g.rotation_direction=1;
    g.use_cone_views=1;
    std::vector<DeviceConeView> views={{make_float3(0,-10,0),make_float3(0,10,0),make_float3(1e-6f,0,0),make_float3(0,0,1e-6f)}};
    Device<DeviceConeView> dv(views);g.cone_views=dv.get();
    Device<unsigned char> labels(std::vector<unsigned char>{1,1,1,2,2,2}),mask(1);
    std::vector<int> map(256,-1);map[1]=0;map[2]=1;
    Device<int> dm(map);Device<float> mu(std::vector<float>{1,2,3,4}),spectrum(std::vector<float>{100,50}),energies(std::vector<float>{40,80});
    Device<float> expected(1),sampled(1),air(1),efficiency(std::vector<float>{.5f,.25f});
    auto run=[&]{
        launchPrimaryKernel<2>(g,labels.get(),dm.get(),mu.get(),spectrum.get(),energies.get(),8000,0,0,123,456,expected.get(),sampled.get(),air.get(),mask.get());sync();
        near(sampled.read()[0],expected.read()[0],0,"disabled photon noise");
        return std::array<float,2>{expected.read()[0],air.read()[0]};
    };
    auto check=[&](double value,double reference,const char* name){const auto x=run();near(x[0],value,std::max(.002,value*2e-6),name);near(x[1],reference,std::max(.002,reference*2e-6),"air expectation");};
    const double value=4000*std::exp(-1.2)+4000*std::exp(-1.8);
    check(value,8000,"two-material two-energy Beer-Lambert");
    require(mask.read()[0]==1,"material hit mask");
    g.samples=8;g.focal_epsilon=.25f;check(value,8000,"oversampling and defocus dose conservation");
    ck(cudaMemcpyToSymbol(c_mu,std::vector<float>{1,2,3,4}.data(),4*sizeof(float)));
    ck(cudaMemcpyToSymbol(c_spectrum,std::vector<float>{100,50}.data(),2*sizeof(float)));
    ck(cudaMemcpyToSymbol(c_energies,std::vector<float>{40,80}.data(),2*sizeof(float)));
    g.use_constant_tables=1;check(value,8000,"constant spectral tables");
    g.detector_response=1;g.detector_efficiency=efficiency.get();
    check(2000*std::exp(-1.2)+1000*std::exp(-1.8),3000,"energy-dependent detector weighting");
    g.detector_response=0;g.geometry_flux=1;
    check(value*2500,8000*2500,"inverse-square fluence, mm-to-meter scale");
    views[0].source.y=-20;views[0].detector_start.y=20;dv.upload(views);
    check(value*625,8000*625,"inverse-square distance ratio");
    g.geometry_flux=0;labels.upload({0,0,0,0,0,0});
    check(8000,8000,"vacuum transmission");require(mask.read()[0]==0,"vacuum mask");
}
}
int main(int argc,char** argv){
    try{
        require(argc==2,"GPU module selection required");ck(cudaSetDevice(0));const std::string test=argv[1];
        if(test=="geometry")geometry();else if(test=="paths")paths();else if(test=="noise")noise();
        else if(test=="response")response();else if(test=="transmission")transmission();
        else if(test=="focal")focal();
        else throw std::runtime_error("unknown GPU module test");
        std::cout<<test<<": PASS\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
