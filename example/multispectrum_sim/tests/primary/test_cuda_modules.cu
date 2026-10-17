#include "TestChecks.hpp"
#include "primary/WarpSpectralTransport.cuh"
#include "primary/DetectorResponse.cuh"
#include <array>
#include <iostream>
#include <vector>
#include <type_traits>
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
__global__ void orbitProbe(DeviceGeometry g,float* out){
    const int views[]={0,g.views_per_turn,2*g.views_per_turn-1};
    for(int i=0;i<3;++i){const auto ray=scannerRay(g,views[i],.5f*(g.nu-1),.5f*(g.nv-1));
        out[i*2]=ray.source.z;out[i*2+1]=ray.detector.z;}
}
void geometry(){
    DeviceGeometry g{};g.phantom_x=1;g.phantom_y=2;g.phantom_z=3;g.phantom_rz=1.57079632679f;
    Device<float> result(16);geometryProbe<<<1,1>>>(g,result.get());sync();auto x=result.read();
    const double expected[]={0,1,0,1,1,1,0,1,0,1,1,1,0,0,0,1};
    for(int i=0;i<16;++i)near(x[i],expected[i],2e-6,"geometry component "+std::to_string(i));
    using namespace yk::spectral;
    for(auto kind:{GeometryKind::FlatCbct,GeometryKind::CylCbct,GeometryKind::FlatHelical,GeometryKind::CylHelical})for(int direction:{-1,1})for(float pitch:{-80.f,80.f}){
        SimulationConfig c;c.geometry.kind=kind;auto& p=c.geometry.parameters;
        p.views=24;p.views_per_turn=12;p.rotation_direction=direction;p.start_angle_rad=.3;
        p.pitch_mm_per_turn=pitch;p.start_z_mm=-120;p.source_offset_z_mm=7;p.offset_v_mm=2;
        const auto d=makeDeviceGeometry(c,1,1);
        const bool helical=kind==GeometryKind::FlatHelical || kind==GeometryKind::CylHelical;
        require(d.pitch==(helical?pitch:0.f),"circular geometry consumed stale helical pitch");
        orbitProbe<<<1,1>>>(d,result.get());sync();const auto actual=result.read();
        const int views[]={0,12,23};
        for(int i=0;i<3;++i){const double z=-120+(helical?direction*double(pitch)*views[i]/12:0);
            near(actual[i*2],z+7,3e-4,"source Z trajectory");near(actual[i*2+1],z+2,3e-4,"detector Z trajectory");}
    }
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
__global__ void outerFocalProbe(DeviceGeometry g,float2* points,int count){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count)return;
    points[p]=sampleFocalPoint(g,makeFocalPartition(g),true,unsigned(p)*0x9e3779b9U);
}
void focalPartitionTests(){
    DeviceGeometry g{};g.normal_fu=2;g.normal_fv=1;g.defocus_fu=4;g.defocus_fv=3;g.focal_epsilon=.25f;
    const auto plan=makeFocalPartition(g);
    near(focalPartitionWeight(plan,true),.25*10/12,1e-7,"outer area exposure");
    near(focalPartitionWeight(plan,false),.75+.25*2/12,1e-7,"shared inner exposure");
    near(focalPartitionWeight(plan,false)+focalPartitionWeight(plan,true),1,1e-7,"partition exposure conservation");
    constexpr int count=65536;Device<float2> points(count);
    for(int shape=0;shape<3;++shape){
        if(shape==1)g.defocus_fu=g.normal_fu;
        if(shape==2){g.defocus_fu=4;g.defocus_fv=g.normal_fv;}
        outerFocalProbe<<<(count+255)/256,256>>>(g,points.get(),count);sync();
        std::array<int,4> bands{};double sx=0,sy=0;
        for(auto q:points.read()){
            require(std::isfinite(q.x)&&std::isfinite(q.y),"nonfinite outer focal point");
            require(std::abs(q.x)<=.5f*g.defocus_fu&&std::abs(q.y)<=.5f*g.defocus_fv,"outer focus bounds");
            require(std::abs(q.x)>=.5f*g.normal_fu||std::abs(q.y)>=.5f*g.normal_fv,"outer sample fell inside normal focus");
            if(q.x<=-.5f*g.normal_fu)++bands[0];else if(q.x>=.5f*g.normal_fu)++bands[1];else if(q.y<0)++bands[2];else ++bands[3];sx+=q.x;sy+=q.y;
        }
        const double area=g.defocus_fu*g.defocus_fv-g.normal_fu*g.normal_fv;
        const double side=.5*(g.defocus_fu-g.normal_fu)*g.defocus_fv/area;
        const double cap=.5*g.normal_fu*(g.defocus_fv-g.normal_fv)/area;
        for(int b=0;b<4;++b)near(double(bands[b])/count,b<2?side:cap,.01,"outer bands not weighted by area");
        near(sx/count,0,.025,"outer U centroid");near(sy/count,0,.025,"outer V centroid");
    }
    g.defocus_fu=g.normal_fu;g.defocus_fv=g.normal_fv;
    near(focalPartitionWeight(makeFocalPartition(g),true),0,0,"equal focus has no outer area");
    g.focal_epsilon=0;near(focalPartitionWeight(makeFocalPartition(g),false),1,0,"zero defocus normal dose");
    g.normal_fu=g.normal_fv=0;g.focal_epsilon=1;
    near(focalPartitionWeight(makeFocalPartition(g),false),0,0,"point inner has zero broad-focus area");
}
void focal(){
    focalPartitionTests();
    yk::spectral::SimulationConfig controls;auto& spot=controls.projection.focal_spot;
    spot.normal_size_u_mm=.4;spot.normal_size_v_mm=.3;spot.defocus_size_u_mm=1;spot.defocus_size_v_mm=1.2;spot.epsilon=.01;
    for(bool normal:{false,true})for(bool defocus:{false,true}){
        spot.enabled=normal;spot.defocus_enabled=defocus;const auto device=makeDeviceGeometry(controls,2,3);
        near(device.normal_fu,normal?.4:0,1e-7,"normal switch device size");
        near(device.normal_fv,normal?.3:0,1e-7,"normal V switch device size");
        near(device.defocus_fu,defocus?1:0,1e-7,"defocus switch device size");
        near(device.focal_epsilon,defocus?.01:0,1e-7,"defocus switch device dose");
    }
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
unsigned hostHash(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);}
float hostUniform(unsigned seed){return (hostHash(seed)&0xffffffU)/16777216.f;}
std::vector<double> directVoxelPaths(DeviceGeometry g,const std::vector<unsigned char>& labels,
    const std::vector<int>& map,float3 source,float3 direction){
    std::vector<double> paths(g.nm);double origin[3]={source.x,source.y,source.z},d[3]={direction.x,direction.y,direction.z};
    double step[3]={g.vx,g.vy,g.vz},low[3]={-.5*g.nx*g.vx,-.5*g.ny*g.vy,-.5*g.nz*g.vz};
    for(int z=0;z<g.nz;++z)for(int y=0;y<g.ny;++y)for(int x=0;x<g.nx;++x){int m=map[labels[(z*g.ny+y)*g.nx+x]];if(m<0||m>=g.nm)continue;
        int cell[3]={x,y,z};double entry=0,exit=1e30;
        for(int axis=0;axis<3;++axis){double lo=low[axis]+cell[axis]*step[axis],hi=lo+step[axis];
            if(std::abs(d[axis])<1e-12){if(origin[axis]<lo||origin[axis]>=hi){exit=-1;break;}}
            else{double a=(lo-origin[axis])/d[axis],b=(hi-origin[axis])/d[axis];if(a>b)std::swap(a,b);entry=std::max(entry,a);exit=std::min(exit,b);}}
        if(exit>entry)paths[m]+=(exit-entry)*.1;
    }return paths;
}
std::array<double,2> multiReference(DeviceGeometry g,int pixel,int view,unsigned seed,
    const std::vector<unsigned char>& labels,const std::vector<int>& map,
    const std::vector<float>& mu,const std::vector<float>& spectrum,const std::vector<float>& energies){
    std::array<double,2> result{};
    // Independent CPU stratification: preserve the broad uniform density by
    // weighting its inner and outer areas, sharing the normal inner samples.
    const bool stratified=g.focal_epsilon>0&&g.defocus_fu>0&&g.defocus_fv>0&&g.normal_fu<=g.defocus_fu&&g.normal_fv<=g.defocus_fv;
    const double area_ratio=stratified?double(g.normal_fu)/g.defocus_fu*g.normal_fv/g.defocus_fv:0;
    const double outer_weight=stratified?g.focal_epsilon*(1-area_ratio):g.focal_epsilon;
    for(int component=0;component<2;++component){double fraction=component?outer_weight:1-outer_weight;if(fraction<=0)continue;
        for(int sample=0;sample<g.samples;++sample){unsigned base=seed^(unsigned(view*g.nu*g.nv+pixel))*0x9e3779b9U^unsigned(sample)*0x85ebca6bU;
            float x=(hostUniform(base+1)-.5f)*(component?g.defocus_fu:g.normal_fu);
            float y=(hostUniform(base+2)-.5f)*(component?g.defocus_fv:g.normal_fv);
            if(component&&stratified){
                const unsigned outer_seed=base^0xd1b54a35U;
                const double u=hostUniform(outer_seed+1),v=hostUniform(outer_seed+2);
                const double vertical=(double(g.defocus_fu)-g.normal_fu)/g.defocus_fu;
                const double horizontal=double(g.normal_fu)/g.defocus_fu*(double(g.defocus_fv)-g.normal_fv)/g.defocus_fv;
                const double position=u*(vertical+horizontal);
                if(vertical>0&&(position<vertical||horizontal<=0)){
                    const double t=2*position/vertical,band=.5*(g.defocus_fu-g.normal_fu);
                    x=float(t<1?-.5*g.defocus_fu+t*band:.5*g.normal_fu+(t-1)*band);y=float((v-.5)*g.defocus_fv);
                }else{
                    const double t=2*(position-vertical)/horizontal,band=.5*(g.defocus_fv-g.normal_fv);
                    x=float((v-.5)*g.normal_fu);y=float(t<1?-.5*g.defocus_fv+t*band:.5*g.normal_fv+(t-1)*band);
                }
            }
            auto ray=scannerRay(g,view,float(pixel%g.nu),float(pixel/g.nu),x,y,hostUniform(base+3)-.5f,hostUniform(base+4)-.5f);
            float3 d=make_float3(ray.detector.x-ray.source.x,ray.detector.y-ray.source.y,ray.detector.z-ray.source.z);
            float length=std::sqrt(d.x*d.x+d.y*d.y+d.z*d.z);d.x/=length;d.y/=length;d.z/=length;
            const auto paths=directVoxelPaths(g,labels,map,phantomInverse(g,ray.source,false),phantomInverse(g,d,true));
            double flux=g.geometry_flux?1000000./(double(length)*length):1;
            for(int e=0;e<g.ne;++e){double exponent=0;for(int m=0;m<g.nm;++m)exponent+=paths[m]*mu[m*g.ne+e];
                double weight=fraction/g.samples*spectrum[e]*flux*energies[e];result[0]+=weight*std::exp(-exponent);result[1]+=weight;}
        }
    }return result;
}
void multisample(){
    DeviceGeometry g{};g.nu=3;g.nv=2;g.nx=5;g.ny=6;g.nz=4;g.vx=.7f;g.vy=1.1f;g.vz=.9f;g.nm=2;g.ne=3;
    g.views_per_turn=13;g.rotation_direction=1;g.sid=15;g.sdd=30;g.pu=.6f;g.pv=.8f;g.geometry_flux=1;
    g.phantom_x=.2f;g.phantom_y=-.3f;g.phantom_z=.1f;g.phantom_rx=.1f;g.phantom_ry=.2f;g.phantom_rz=-.12f;
    g.normal_fu=.4f;g.normal_fv=.2f;g.defocus_fu=1;g.defocus_fv=.9f;g.focal_epsilon=.25f;
    std::vector<unsigned char> labels(120);for(int z=0;z<4;++z)for(int y=0;y<6;++y)for(int x=0;x<5;++x)labels[(z*6+y)*5+x]=((x+y+z)%4==0)?0:(x<2?1:2);
    std::vector<int> map(256,-1);map[1]=0;map[2]=1;
    std::vector<float> mu={.2f,.4f,.7f,1,.8f,.5f},sw={300,120,50},en={30,60,90};
    Device<unsigned char> dl(labels),mask(6);Device<int> dm(map);Device<float> dmu(mu),ds(sw),de(en),expected(6),sampled(6),air(6);Device<unsigned long long> counters(8);
    for(int samples:{1,2,8,30,64}){g.samples=samples;
        launchPrimaryKernel<2>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),20700,0,2,123,456,expected.get(),sampled.get(),air.get(),mask.get());sync();auto actual=expected.read(),referenceAir=air.read();
        for(int pixel=0;pixel<6;++pixel){auto reference=multiReference(g,pixel,2,123,labels,map,mu,sw,en);
            near(actual[pixel],reference[0],std::max(.02,reference[0]*3e-5),"heterogeneous multi-ray nonlinear transmission");
            near(referenceAir[pixel],reference[1],std::max(.02,reference[1]*3e-5),"multi-ray dose averaging");}
        ck(cudaMemset(counters.get(),0,8*sizeof(unsigned long long)));
        launchPrimaryKernel<2>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),20700,0,2,123,456,expected.get(),sampled.get(),air.get(),mask.get(),1,counters.get());sync();auto profiled=expected.read();
        for(int p=0;p<6;++p)near(profiled[p],actual[p],std::max(.02,double(actual[p])*2e-6),"profiling changed primary output");
    }
    g.samples=30;
    ck(cudaMemset(counters.get(),0,8*sizeof(unsigned long long)));
    launchPrimaryKernel<2>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),20700,0,0,123,456,expected.get(),sampled.get(),air.get(),mask.get(),1,counters.get());sync();
    const auto measured=counters.read();require(measured[4]>0&&measured[5]==60*measured[4],"instrumented ray count disagrees with focal-component sampling");
    require(measured[6]<=measured[5]&&measured[7]>=measured[6],"instrumented traversal counts are inconsistent");
    // Real scanner distances also exercise entry-point cancellation and voxel edges.
    g.sid=709;g.sdd=1143;
    launchPrimaryKernel<2>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),20700,0,2,123,456,expected.get(),sampled.get(),air.get(),mask.get());sync();
    auto farExpected=expected.read();
    for(int p=0;p<6;++p){auto reference=multiReference(g,p,2,123,labels,map,mu,sw,en);near(farExpected[p],reference[0],std::max(.002,reference[0]*4e-5),"multi-ray traversal at 709/1143 scanner distances");}
    // Exercise both counter capacities at their boundary, with energy-dependent
    // response, quantum noise, inverse-square flux and two focal components.
    g.samples=8;
    for(int bins:{119,128,129,256}){
        g.ne=bins;
        std::vector<float> wideMu(2*bins),wideSpectrum(bins),wideEnergy(bins),response(6*bins);
        for(int e=0;e<bins;++e){wideEnergy[e]=20.f+e*.4f;wideSpectrum[e]=50.f+e%7;
            wideMu[e]=.2f+.001f*e;wideMu[bins+e]=.7f+.002f*e;
            for(int p=0;p<6;++p)response[p*bins+e]=.2f+.05f*(e%5)+.01f*p;}
        Device<float> wm(wideMu),ws(wideSpectrum),we(wideEnergy),wr(response);
        g.detector_response=0;
        launchPrimaryKernel<2>(g,dl.get(),dm.get(),wm.get(),ws.get(),we.get(),1,0,2,123,456,expected.get(),sampled.get(),air.get(),mask.get());sync();
        auto wideActual=expected.read();
        for(int p=0;p<6;++p){auto ref=multiReference(g,p,2,123,labels,map,wideMu,wideSpectrum,wideEnergy);
            near(wideActual[p],ref[0],std::max(.02,ref[0]*4e-5),"counter capacity boundary CPU reference");}
        g.detector_response=1;g.detector_efficiency=wr.get();
        launchPrimaryKernel<2>(g,dl.get(),dm.get(),wm.get(),ws.get(),we.get(),1,1,2,123,456,expected.get(),sampled.get(),air.get(),mask.get());sync();
        auto narrowExpected=expected.read(),narrowSampled=sampled.read(),narrowAir=air.read();auto narrowMask=mask.read();
        primaryKernel<2,false,256><<<1,256>>>(g,dl.get(),dm.get(),wm.get(),ws.get(),we.get(),1,1,2,123,456,expected.get(),sampled.get(),air.get(),mask.get());sync();
        auto fullExpected=expected.read(),fullSampled=sampled.read(),fullAir=air.read();
        require(narrowExpected==fullExpected&&narrowSampled==fullSampled&&narrowAir==fullAir&&narrowMask==mask.read(),"counter capacity changed noisy response outputs");
    }
    g.detector_response=0;g.detector_efficiency=nullptr;
    // The generic 32-slot path must read only nm actual attenuation rows.
    g.nm=11;g.ne=2;g.samples=8;mu.resize(22);for(int i=0;i<22;++i)mu[i]=.05f*(i+1);
    sw={100,50};en={40,80};for(size_t p=0;p<labels.size();++p)labels[p]=static_cast<unsigned char>(p%11+1);
    map.assign(256,-1);for(int m=0;m<11;++m)map[m+1]=m;
    Device<unsigned char> genericLabels(labels);Device<int> genericMap(map);Device<float> genericMu(mu),genericSpectrum(sw),genericEnergy(en);
    launchPrimaryKernel<32>(g,genericLabels.get(),genericMap.get(),genericMu.get(),genericSpectrum.get(),genericEnergy.get(),8000,0,2,123,456,expected.get(),sampled.get(),air.get(),mask.get());sync();auto actual=expected.read();
    for(int p=0;p<6;++p){auto reference=multiReference(g,p,2,123,labels,map,mu,sw,en);near(actual[p],reference[0],std::max(.02,reference[0]*3e-5),"generic attenuation rows exceeded actual material count");}
}

template<int M>
void launchWarpPrimaryTest(int option,const DeviceGeometry& g,const unsigned char* labels,
    const int* map,const float* mu,const float* spectrum,const float* energies,
    int quantum_noise,int view,unsigned seed,unsigned quantum_seed,float* expected,
    float* sampled,float* air,unsigned char* mask,int views=1){
    if constexpr(M==2||M==10){
        if(option){launchWarpPrimarySized<M,128,true>(g,labels,map,mu,spectrum,energies,quantum_noise,view,seed,quantum_seed,expected,sampled,air,mask,views,false);return;}
    }
    launchWarpPrimaryKernel<M>(g,labels,map,mu,spectrum,energies,quantum_noise,view,seed,quantum_seed,expected,sampled,air,mask,views);
}
// Independent warp kernel versus scalar projector and double CPU box reference.
void warpProfile(){
    DeviceGeometry g{};g.nu=3;g.nv=2;g.nx=5;g.ny=6;g.nz=4;g.vx=.7f;g.vy=1.1f;g.vz=.9f;
    g.nm=53;g.views=2;g.views_per_turn=8;g.sid=100;g.sdd=160;g.pu=.4f;g.pv=.5f;g.rotation_direction=1;
    g.normal_fu=.1f;g.normal_fv=.2f;g.defocus_fu=.4f;g.defocus_fv=.6f;g.focal_epsilon=.25f;
    std::vector<unsigned char> labels(120);for(int i=0;i<120;++i)labels[i]=i%53;
    std::vector<int> map(256,-1);for(int m=0;m<53;++m)map[m]=m;
    Device<unsigned char> dl(labels),mask(12);Device<int> dm(map);
    Device<float> expected(12),sampled(12),air(12);Device<unsigned long long> counters(WarpProfileCounterCount);
    for(int bins:{119,238}){
        g.ne=bins;std::vector<float> mu(53*bins),sw(bins,50),en(bins);
        for(size_t i=0;i<mu.size();++i)mu[i]=.1f+.0002f*i;
        for(int e=0;e<bins;++e)en[e]=20.f+e*.4f;
        Device<float> dmu(mu),ds(sw),de(en);
        for(int samples:{1,4,33})for(int noise:{0,1}){
            g.samples=samples;
            if(bins<=128)launchWarpPrimarySized<64,128>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),noise,0,123,456,expected.get(),sampled.get(),air.get(),mask.get(),2,false);
            else launchWarpPrimarySized<64,256>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),noise,0,123,456,expected.get(),sampled.get(),air.get(),mask.get(),2,false);
            sync();
            const auto ref=expected.read(),noisy=sampled.read(),refair=air.read();const auto refmask=mask.read();
            if(bins<=128)launchWarpPrimarySized<64,128,false,false,true>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),noise,0,123,456,expected.get(),sampled.get(),air.get(),mask.get(),2,false);
            else launchWarpPrimarySized<64,256,false,false,true>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),noise,0,123,456,expected.get(),sampled.get(),air.get(),mask.get(),2,false);
            sync();require(refmask==mask.read(),"shared material paths changed object mask");
            const auto actual=expected.read(),actualNoise=sampled.read(),actualAir=air.read();
            for(size_t p=0;p<ref.size();++p){
                near(actual[p],ref[p],std::max(.001,double(ref[p])*2e-6),"shared paths energy reduction");
                near(actualNoise[p],noisy[p],std::max(.001,double(noisy[p])*2e-6),"shared paths noise reduction");
                near(actualAir[p],refair[p],std::max(.001,double(refair[p])*2e-6),"shared paths air reduction");
            }
            ck(cudaMemset(counters.get(),0,WarpProfileCounterCount*sizeof(unsigned long long)));
            if(bins<=128)launchWarpPrimarySized<64,128>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),noise,0,123,456,expected.get(),sampled.get(),air.get(),mask.get(),2,false,counters.get());
            else launchWarpPrimarySized<64,256>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),noise,0,123,456,expected.get(),sampled.get(),air.get(),mask.get(),2,false,counters.get());
            sync();require(ref==expected.read()&&noisy==sampled.read()&&refair==air.read()&&refmask==mask.read(),"warp profiling changed numerical output");
            const auto counts=counters.read();
            require(counts[6]>0&&counts[7]==counts[6]*samples*2,"warp sampled ray count differs from focal components");
            require(counts[8]<=counts[7]&&counts[9]>=counts[8],"warp traversal counters are inconsistent");
            for(int stage=0;stage<6;++stage)require(counts[stage]>0,"empty warp stage timing");
        }
    }
}
void warpPolicy(){
    for(int nm:{1,2,10,11,32,33,53,64})for(int ne:{3,119,128,129,238,256})for(int rays:{1,2,3,4,5,8,9,16,17,32,50}){
        const auto layout=selectPrimaryLaunchLayout(nm,ne,rays);
        const int group=ne>128?32:(rays<=2||(nm<=32&&rays<=4)?4:(rays<=8?8:(nm<=32&&rays>16?32:16)));
        require(layout.threads_per_pixel==group,"primary launch group selection");
        require(layout.threads_per_block==(group==16&&nm>32?64:128),"primary launch block selection");
    }
    const std::array<std::array<int,3>,5> invalidDims{{{0,119,4},{65,119,4},{10,0,4},{10,257,4},{10,119,0}}};
    for(const auto& dims:invalidDims){
        bool rejected=false;try{selectPrimaryLaunchLayout(dims[0],dims[1],dims[2]);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"invalid launch dimensions accepted");
    }
    DeviceGeometry g{};g.nu=3;g.nv=2;g.nx=5;g.ny=6;g.nz=4;g.vx=.7f;g.vy=1.1f;g.vz=.9f;
    g.views=2;g.views_per_turn=8;g.sid=100;g.sdd=160;g.pu=.4f;g.pv=.5f;g.rotation_direction=1;g.geometry_flux=1;
    g.phantom_x=.2f;g.phantom_y=-.3f;g.phantom_z=.1f;g.phantom_rx=.1f;g.phantom_ry=.2f;g.phantom_rz=-.12f;
    g.normal_fu=.1f;g.normal_fv=.2f;g.defocus_fu=.4f;g.defocus_fv=.6f;g.focal_epsilon=.25f;
    Device<unsigned char> dl(120),mask(12);Device<int> dm(256);Device<float> expected(12),sampled(12),air(12);
    Device<unsigned long long> counters(WarpProfileCounterCount);
    for(int nm:{1,2,10,11,32,33,53,64}){
        g.nm=nm;std::vector<unsigned char> labels(120);for(int i=0;i<120;++i)labels[i]=i%7?i%nm:255;
        std::vector<int> map(256,-1);for(int m=0;m<nm;++m)map[m]=m;dl.upload(labels);dm.upload(map);
        for(int bins:{3,119,129,238,256}){
            g.ne=bins;std::vector<float> mu(nm*bins),sw(bins,50),en(bins);
            std::vector<float> efficiency(6*bins);for(size_t i=0;i<efficiency.size();++i)efficiency[i]=.2f+.02f*(i%13);
            Device<float> dr(efficiency);g.detector_efficiency=dr.get();
            for(size_t i=0;i<mu.size();++i)mu[i]=.1f+.00002f*i;
            for(int e=0;e<bins;++e)en[e]=20.f+e*.4f;
            Device<float> dmu(mu),ds(sw),de(en);
            auto launch=[&](int first,int views,float* e,float* s,float* a,unsigned char* h,unsigned long long* profile=nullptr){
                if(nm<=10)launchWarpPrimaryKernel<10>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),g.detector_response,first,123,456,e,s,a,h,views,false,profile);
                else if(nm<=32)launchWarpPrimaryKernel<32>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),g.detector_response,first,123,456,e,s,a,h,views,false,profile);
                else launchWarpPrimaryKernel<64>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),g.detector_response,first,123,456,e,s,a,h,views,false,profile);
            };
            for(int rays:{1,2,3,4,5,8,9,16,17,32,50})for(int noise:{0,1}){
                g.samples=rays;g.detector_response=noise;
                if(bins<=128)launchWarpPrimarySized<64,128>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),noise,0,123,456,expected.get(),sampled.get(),air.get(),mask.get(),2,false);
                else launchWarpPrimarySized<64,256>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),noise,0,123,456,expected.get(),sampled.get(),air.get(),mask.get(),2,false);
                sync();const auto ref=expected.read(),refNoise=sampled.read(),refAir=air.read();const auto refMask=mask.read();
                launch(0,2,expected.get(),sampled.get(),air.get(),mask.get());sync();
                const auto actual=expected.read(),actualNoise=sampled.read(),actualAir=air.read();
                require(refMask==mask.read(),"adaptive launch mask mismatch");
                for(size_t i=0;i<ref.size();++i){
                    near(actual[i],ref[i],std::max(.001,double(ref[i])*2e-6),"adaptive expected energy");
                    near(actualNoise[i],refNoise[i],std::max(.001,double(refNoise[i])*2e-6),"adaptive quantum energy");
                    near(actualAir[i],refAir[i],std::max(.001,double(refAir[i])*2e-6),"adaptive air energy");
                }
                for(int first=0;first<2;++first)launch(first,1,expected.get()+first*6,sampled.get()+first*6,air.get()+first*6,mask.get()+first*6);
                sync();require(actual==expected.read()&&actualNoise==sampled.read()&&actualAir==air.read()&&refMask==mask.read(),"adaptive batching changed output");
                // Check instrumented specialization on every selected layout,
                // including padded material capacities and detector response.
                if(bins==119&&noise){
                    ck(cudaMemset(counters.get(),0,WarpProfileCounterCount*sizeof(unsigned long long)));
                    launch(0,2,expected.get(),sampled.get(),air.get(),mask.get(),counters.get());sync();
                    require(actual==expected.read()&&actualNoise==sampled.read()&&actualAir==air.read()&&refMask==mask.read(),"adaptive instrumentation changed output");
                    const auto count=counters.read();require(count[6]>0&&count[7]==count[6]*rays*2,"adaptive sampled ray counts");
                }
            }
        }
    }
}
void warp(bool experiments=false){
    DeviceGeometry g{};g.nu=3;g.nv=2;g.nx=5;g.ny=6;g.nz=4;g.vx=.7f;g.vy=1.1f;g.vz=.9f;
    g.views_per_turn=13;g.rotation_direction=1;g.sid=709;g.sdd=1143;g.pu=.6f;g.pv=.8f;g.geometry_flux=1;
    g.phantom_x=.2f;g.phantom_y=-.3f;g.phantom_z=.1f;g.phantom_rx=.1f;g.phantom_ry=.2f;g.phantom_rz=-.12f;
    g.normal_fu=.4f;g.normal_fv=.2f;g.defocus_fu=1;g.defocus_fv=.9f;
    std::vector<unsigned char> labels(120);for(int p=0;p<120;++p)labels[p]=p%4==0?0:(p%2+1);
    std::vector<int> map(256,-1);map[1]=0;map[2]=1;
    Device<unsigned char> dl(labels),mask(12);Device<int> dm(map);
    Device<float> expected(12),sampled(12),air(12);
    auto verify=[&](auto slots,const std::vector<float>& mu,const std::vector<float>& sw,const std::vector<float>& en,int option=0){
        constexpr int M=decltype(slots)::value;
        Device<float> dmu(mu),ds(sw),de(en);
        std::vector<float> efficiency(6*g.ne);for(size_t p=0;p<efficiency.size();++p)efficiency[p]=.2f+.02f*(p%13);
        Device<float> dr(efficiency);g.detector_efficiency=dr.get();
        for(int noise:{0,1}){
            g.detector_response=noise;
            launchPrimaryKernel<M>(g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),1,noise,2,123,456,expected.get(),sampled.get(),air.get(),mask.get(),2);sync();
            auto refExpected=expected.read(),refSampled=sampled.read(),refAir=air.read();auto refMask=mask.read();
            launchWarpPrimaryTest<M>(option,g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),noise,2,123,456,expected.get(),sampled.get(),air.get(),mask.get(),2);sync();
            auto actual=expected.read(),noisy=sampled.read(),actualAir=air.read();
            require(mask.read()==refMask,"warp object mask differs");
            for(int p=0;p<12;++p){
                near(actual[p],refExpected[p],std::max(.01,double(refExpected[p])*3e-5),"warp expected energy differs");
                near(noisy[p],refSampled[p],std::max(.01,double(refSampled[p])*3e-5),"warp quantum sample differs");
                near(actualAir[p],refAir[p],std::max(.01,double(refAir[p])*2e-4),"warp air energy differs");
                if(!noise){auto cpu=multiReference(g,p%6,2+p/6,123,labels,map,mu,sw,en);
                    near(actual[p],cpu[0],std::max(.01,cpu[0]*4e-5),"warp independent CPU transmission");
                    near(actualAir[p],cpu[1],std::max(.01,cpu[1]*4e-5),"warp independent CPU air");}
            }
            auto repeated=noisy;
            launchWarpPrimaryTest<M>(option,g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),noise,2,123,456,expected.get(),sampled.get(),air.get(),mask.get(),2);sync();
            require(repeated==sampled.read(),"warp output is not deterministic");
            for(int v=0;v<2;++v)launchWarpPrimaryTest<M>(option,g,dl.get(),dm.get(),dmu.get(),ds.get(),de.get(),noise,2+v,123,456,expected.get()+v*6,sampled.get()+v*6,air.get()+v*6,mask.get()+v*6);
            sync();require(repeated==sampled.read(),"warp batch size changed output");
        }
    };
    g.nm=2;
    for(int bins:{3,119,128,129,256}){
        g.ne=bins;std::vector<float> mu(2*bins),sw(bins),en(bins);
        for(int e=0;e<bins;++e){mu[e]=.2f+.001f*e;mu[bins+e]=.7f+.002f*e;sw[e]=50.f+e%7;en[e]=20.f+.4f*e;}
        for(int samples:{1,2,8,30,31,32,33,64}){g.samples=samples;g.focal_epsilon=.25f;verify(std::integral_constant<int,2>{},mu,sw,en);if(experiments&&bins<=128)for(int option:{1})verify(std::integral_constant<int,2>{},mu,sw,en,option);}
        // Single focal component and a vacuum volume, including epsilon endpoints.
        g.samples=33;g.focal_epsilon=0;verify(std::integral_constant<int,2>{},mu,sw,en);
        dl.upload(std::vector<unsigned char>(120));g.focal_epsilon=1;auto saved=labels;labels.assign(120,0);
        verify(std::integral_constant<int,2>{},mu,sw,en);labels=saved;dl.upload(labels);
    }
    // 10-material production specialization and 11-material generic fallback.
    for(int materials:{10,11}){
        g.nm=materials;g.ne=119;g.samples=30;g.focal_epsilon=.01f;
        map.assign(256,-1);for(int m=0;m<materials;++m)map[m+1]=m;
        for(int p=0;p<120;++p)labels[p]=p%materials+1;dl.upload(labels);dm.upload(map);
        std::vector<float> mu(materials*g.ne),sw(g.ne,100),en(g.ne);
        for(size_t i=0;i<mu.size();++i)mu[i]=.1f+.0002f*i;
        for(int e=0;e<g.ne;++e)en[e]=20.f+e*.4f;
        if(materials==10){verify(std::integral_constant<int,10>{},mu,sw,en);if(experiments)for(int option:{1})verify(std::integral_constant<int,10>{},mu,sw,en,option);}
        else verify(std::integral_constant<int,32>{},mu,sw,en);
    }
}

}
int main(int argc,char** argv){
    try{
        require(argc==2,"GPU module selection required");ck(cudaSetDevice(0));const std::string test=argv[1];
        if(test=="geometry")geometry();else if(test=="paths")paths();else if(test=="noise")noise();
        else if(test=="response")response();else if(test=="transmission")transmission();
        else if(test=="focal")focal();else if(test=="multisample")multisample();else if(test=="warp")warp();else if(test=="warp_coefficients")warp(true);else if(test=="warp_profile")warpProfile();else if(test=="warp_policy")warpPolicy();
        else throw std::runtime_error("unknown GPU module test");
        std::cout<<test<<": PASS\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
