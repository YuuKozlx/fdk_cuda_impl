#include "HuberTvLaunch.cuh"
#include <algorithm>
#include <cmath>
namespace YK::Iter {
namespace {
    __device__ float read(const float* x, int i, int j, int k, int nx, int ny) { return x[(size_t(k) * ny + j) * nx + i]; }
    __device__ float flux(const float* x, int i, int j, int k, int nx, int ny, int nz, float hx, float hy, float hz, float delta, bool volume, int axis)
    {
        const float c = read(x, i, j, k, nx, ny);
        const float gx = i + 1 < nx ? (read(x, i + 1, j, k, nx, ny) - c) / hx : 0.f;
        const float gy = j + 1 < ny ? (read(x, i, j + 1, k, nx, ny) - c) / hy : 0.f;
        const float gz = volume && k + 1 < nz ? (read(x, i, j, k + 1, nx, ny) - c) / hz : 0.f;
        return (axis == 0 ? gx : axis == 1 ? gy
                                           : gz)
            / fmaxf(delta, sqrtf(gx * gx + gy * gy + gz * gz));
    }
    __global__ void gradient(const float* x, float* out, int nx, int ny, int nz, float hx, float hy, float hz, float delta, bool volume, size_t n)
    {
        for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < n; index += size_t(gridDim.x) * blockDim.x) {
            const int i = int(index % nx), j = int((index / nx) % ny), k = int(index / (size_t(nx) * ny));
            float g = -flux(x, i, j, k, nx, ny, nz, hx, hy, hz, delta, volume, 0) / hx;
            g -= flux(x, i, j, k, nx, ny, nz, hx, hy, hz, delta, volume, 1) / hy;
            if (i > 0)
                g += flux(x, i - 1, j, k, nx, ny, nz, hx, hy, hz, delta, volume, 0) / hx;
            if (j > 0)
                g += flux(x, i, j - 1, k, nx, ny, nz, hx, hy, hz, delta, volume, 1) / hy;
            if (volume) {
                g -= flux(x, i, j, k, nx, ny, nz, hx, hy, hz, delta, volume, 2) / hz;
                if (k > 0)
                    g += flux(x, i, j, k - 1, nx, ny, nz, hx, hy, hz, delta, volume, 2) / hz;
            }
            out[index] = g;
        }
    }
    __global__ void update(float* u, const float* z, const float* g, size_t n, float gamma, float eta, bool positive)
    {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
            const float value = u[i] - eta * (u[i] - z[i] + gamma * g[i]);
            u[i] = positive ? fmaxf(0.f, value) : value;
        }
    }
    __global__ void boundedUpdate(float* u, const float* z, const float* g, size_t n, float gamma, float eta, float lo, float hi)
    {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
            u[i] = fminf(hi, fmaxf(lo, u[i] - eta * (u[i] - z[i] + gamma * g[i])));
    }

    __global__ void stationarity(const float* u,const float* z,const float* g,size_t n,float gamma,float lo,float hi,double* partial){
        __shared__ double sum[256];double value=0;
        for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x){
            const float grad=u[i]-z[i]+gamma*g[i];
            const float projected=fminf(hi,fmaxf(lo,u[i]-grad));
            const double r=double(u[i])-projected;value+=r*r;
        }
        sum[threadIdx.x]=value;__syncthreads();
        for(unsigned stride=128;stride;stride>>=1){if(threadIdx.x<stride)sum[threadIdx.x]+=sum[threadIdx.x+stride];__syncthreads();}
        if(threadIdx.x==0)partial[blockIdx.x]=sum[0];
    }
    __global__ void dualUpdate(const float* x, float* p, int nx, int ny, int nz, float hx, float hy, float hz, int dim, float sigma, float gamma, size_t n)
    {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
            int ix = int(i % nx), iy = int((i / nx) % ny), iz = int(i / (size_t(nx) * ny));
            float a = ix + 1 < nx ? p[i] + sigma * (x[i + 1] - x[i]) / hx : 0;
            float b = iy + 1 < ny ? p[n + i] + sigma * (x[i + nx] - x[i]) / hy : 0;
            float c = dim == 3 && iz + 1 < nz ? p[2 * n + i] + sigma * (x[i + size_t(nx) * ny] - x[i]) / hz : 0;
            float scale = fmaxf(1.f, sqrtf(a * a + b * b + c * c) / gamma);
            p[i] = a / scale;
            p[n + i] = b / scale;
            if (dim == 3)
                p[2 * n + i] = c / scale;
        }
    }
    __global__ void primalUpdate(float* u, float* bar, const float* z, const float* p, int nx, int ny, int nz, float hx, float hy, float hz, int dim, float tau, float lo, float hi, size_t n)
    {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
            int ix = int(i % nx), iy = int((i / nx) % ny), iz = int(i / (size_t(nx) * ny));
            float dt = -p[i] / hx - p[n + i] / hy;
            if (ix)
                dt += p[i - 1] / hx;
            if (iy)
                dt += p[n + i - nx] / hy;
            if (dim == 3) {
                dt -= p[2 * n + i] / hz;
                if (iz)
                    dt += p[2 * n + i - size_t(nx) * ny] / hz;
            }
            float old = u[i];
            float next = fminf(hi, fmaxf(lo, (old - tau * dt + tau * z[i]) / (1 + tau)));
            u[i] = next;
            bar[i] = 2 * next - old;
        }
    }

}
void huber_tv_gradient_launch(const float* x, float* g, int nx, int ny, int nz, float hx, float hy, float hz, float delta, int dimensions, cudaStream_t stream)
{
    const size_t n = size_t(nx) * ny * nz;
    gradient<<<unsigned(std::min(size_t(65535), (n + 255) / 256)), 256, 0, stream>>>(x, g, nx, ny, nz, hx, hy, hz, delta, dimensions == 3, n);
}
void huber_tv_prox_update_launch(float* u, const float* z, const float* g, size_t n, float gamma, float eta, bool positive, cudaStream_t stream)
{
    update<<<unsigned(std::min(size_t(65535), (n + 255) / 256)), 256, 0, stream>>>(u, z, g, n, gamma, eta, positive);
}

void bounded_prox_update_launch(float* u, const float* z, const float* g, size_t n, float gamma, float eta, float lo, float hi, cudaStream_t s)
{
    boundedUpdate<<<unsigned(std::min(size_t(65535), (n + 255) / 256)), 256, 0, s>>>(u, z, g, n, gamma, eta, lo, hi);
}
void prox_stationarity_launch(const float* u,const float* z,const float* g,size_t n,float gamma,float lo,float hi,double* partial,cudaStream_t stream){
    stationarity<<<1024,256,0,stream>>>(u,z,g,n,gamma,lo,hi,partial);
}
void tv_dual_launch(const float* x, float* p, int nx, int ny, int nz, float hx, float hy, float hz, int dim, float sigma, float gamma, cudaStream_t s)
{
    size_t n = size_t(nx) * ny * nz;
    dualUpdate<<<unsigned(std::min(size_t(65535), (n + 255) / 256)), 256, 0, s>>>(x, p, nx, ny, nz, hx, hy, hz, dim, sigma, gamma, n);
}
void tv_primal_launch(float* u, float* bar, const float* z, const float* p, int nx, int ny, int nz, float hx, float hy, float hz, int dim, float tau, float lo, float hi, cudaStream_t s)
{
    size_t n = size_t(nx) * ny * nz;
    primalUpdate<<<unsigned(std::min(size_t(65535), (n + 255) / 256)), 256, 0, s>>>(u, bar, z, p, nx, ny, nz, hx, hy, hz, dim, tau, lo, hi, n);
}
}
