#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

struct HoloState {
    float radius;          // holographic radius
    float mass;            // holographic mass (to be updated)
    float pressure;        // local ether pressure
};

// kernel that updates holographic mass for an array of objects
__global__ void updateHolographicMass(HoloState* d_state, int n);