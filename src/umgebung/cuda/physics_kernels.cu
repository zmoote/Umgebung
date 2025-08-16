#include "umgebung/cuda/physics_kernels.h"
#include <cuda_fp16.h>   // if you need half‑precision

// Simple placeholder: M = (4/3)*pi*p*R^3  (p from ether pressure)
__global__ void updateHolographicMass(HoloState* d_state, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float PI = 3.1415926535f;
    const float R = d_state[i].radius;
    const float rho = d_state[i].pressure;      // treat pressure as energy density

    d_state[i].mass = (4.f/3.f) * PI * rho * R*R*R;
}

class GPUPhysics {
public:
    GPUPhysics() {
        // Allocate device buffer for 1000 bodies (example)
        cudaMalloc(&d_state_, 1000 * sizeof(HoloState));
        nBodies_ = 1000;
    }

    ~GPUPhysics() { cudaFree(d_state_); }

    void update(float dt) {
        int threads = 256;
        int blocks  = (nBodies_ + threads - 1) / threads;
        updateHolographicMass<<<blocks, threads>>>(d_state_, nBodies_);
        cudaDeviceSynchronize();
    }

    HoloState* deviceState() const { return d_state_; }

private:
    HoloState* d_state_;
    int nBodies_;
};