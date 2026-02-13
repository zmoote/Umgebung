#pragma once

#include "umgebung/renderer/gl/Shader.hpp"
#include "umgebung/renderer/Camera.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <vector>

// Forward declare for CUDA resource
struct cudaGraphicsResource;

namespace Umgebung::renderer
{

    class DebugRenderer
    {
    public:
        void init();
        void shutdown();

        void beginFrame(const Camera& camera);
        void endFrame();

        void drawBox(const glm::mat4& transform, const glm::vec4& color);
        void drawSphere(const glm::mat4& transform, const glm::vec4& color);
        
        // --- Particle Rendering ---
        // Initializes the VBO for a certain capacity and registers it with CUDA
        void initParticles(size_t initialCapacity);
        
        // Renders the current particles in the VBO
        void drawParticles(const glm::vec4& color);
        
        // Returns the CUDA resource for mapping in the physics system
        cudaGraphicsResource* getParticleCudaResource();
        
        // Updates the number of particles to draw
        void setParticleCount(size_t count);


    private:
        std::unique_ptr<gl::Shader> shader_;
        
        // Cube resources
        unsigned int cubeVAO_ = 0, cubeVBO_ = 0;
        
        // Sphere resources
        unsigned int sphereVAO_ = 0, sphereVBO_ = 0, sphereEBO_ = 0;
        unsigned int sphereIndexCount_ = 0;
        
        // Particle resources for CUDA-GL Interop
        unsigned int particleVAO_ = 0;
        unsigned int particleVBO_ = 0;
        cudaGraphicsResource* particleCudaResource_ = nullptr;
        size_t particleCount_ = 0;
        size_t particleCapacity_ = 0;


        void setupCube();
        void setupSphere();
    };

} // namespace Umgebung::renderer
