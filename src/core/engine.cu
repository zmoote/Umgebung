#include "umgebung/engine.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

namespace umgebung {

    __global__ void generate_flower_kernel(Point3D* d_centers, int total_points, double r, int global_offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x + global_offset;
        if (idx >= total_points) return;
        
        int local_idx = idx - global_offset;

        if (idx == 0) {
            d_centers[local_idx].x = 0.0;
            d_centers[local_idx].y = 0.0;
            d_centers[local_idx].z = 0.0;
            return;
        }

        // Find the ring level `l` and index `i` on that ring
        int l = floor((1.0 + sqrt(1.0 + 4.0 * (idx - 1) / 3.0)) / 2.0);
        
        // Correct potential floating-point inaccuracies
        int offset = 1 + 3 * l * (l - 1);
        if (idx < offset) {
            l--;
            offset = 1 + 3 * l * (l - 1);
        } else if (idx >= 1 + 3 * (l + 1) * l) {
            l++;
            offset = 1 + 3 * l * (l - 1);
        }
        
        int i = idx - offset;
        int num_points = l * 6;
        
        double angle = i * (2.0 * constants::PI / num_points);
        double dist = l * 2.0 * r;
        
        d_centers[local_idx].x = dist * cos(angle);
        d_centers[local_idx].y = dist * sin(angle);
        d_centers[local_idx].z = 0.0;
    }

    void FlowerOfLife::generate(int levels) {
        units.clear();
        if (levels < 0) return;

        // Calculate total points required. Use long long to avoid overflow on massive levels.
        long long total_points_ll = 1LL + 3LL * levels * (levels + 1LL);
        // Cast back to int since our kernel uses ints for now. 
        // Note: For true infinite scaling, we'd upgrade kernel indices to size_t.
        int total_points = static_cast<int>(total_points_ll); 
        
        units.reserve(total_points);

        // Auto-Tuned Chunking: Check available VRAM
        size_t free_byte;
        size_t total_byte;
        cudaError_t err = cudaMemGetInfo(&free_byte, &total_byte);
        if (err != cudaSuccess) {
            std::cerr << "CUDA MemGetInfo failed: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        // Reserve 80% of free VRAM for our chunk buffer to be safe,
        // but cap it at ~5 million points per chunk to prevent Windows TDR (Timeout Detection and Recovery) timeouts
        size_t max_points_by_vram = (free_byte * 0.8) / sizeof(Point3D);
        int chunk_size = std::min(static_cast<int>(max_points_by_vram), 5000000);
        chunk_size = std::min(chunk_size, total_points);

        Point3D* d_centers = nullptr;
        err = cudaMalloc(&d_centers, chunk_size * sizeof(Point3D));
        if (err != cudaSuccess) {
            std::cerr << "CUDA Malloc failed for chunk size: " << chunk_size << " - " << cudaGetErrorString(err) << std::endl;
            return;
        }

        double r = PSU::radius();
        int threadsPerBlock = 256;

        std::vector<Point3D> h_centers(chunk_size);

        for (int global_offset = 0; global_offset < total_points; global_offset += chunk_size) {
            int current_chunk_size = std::min(chunk_size, total_points - global_offset);
            int blocksPerGrid = (current_chunk_size + threadsPerBlock - 1) / threadsPerBlock;

            generate_flower_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_centers, total_points, r, global_offset);
            
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::cerr << "CUDA Kernel failed at offset " << global_offset << ": " << cudaGetErrorString(err) << std::endl;
                break;
            }

            cudaMemcpy(h_centers.data(), d_centers, current_chunk_size * sizeof(Point3D), cudaMemcpyDeviceToHost);

            // Populate units from host array for this chunk
            for (int i = 0; i < current_chunk_size; ++i) {
                units.emplace_back(h_centers[i].x, h_centers[i].y, h_centers[i].z);
            }
        }

        cudaFree(d_centers);
    }

} // namespace umgebung
