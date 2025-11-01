#pragma once

#include <memory>
#include <string>

// --- Forward Declarations ---
// We let the includes below handle glm types
namespace Umgebung::renderer {
    class Mesh;
    class Camera;
    namespace gl { class Shader; } // <-- Correctly namespace Shader
}
namespace Umgebung::asset {
    class ModelLoader;
}
// Include headers that define types we use
#include "umgebung/renderer/Camera.hpp"
#include "umgebung/renderer/Mesh.hpp"
// ---

namespace Umgebung::renderer {

    class Renderer {
    public:
        Renderer();
        ~Renderer();

        void init();
        void shutdown();

        gl::Shader& getShader(); // <-- Use gl::Shader
        Camera& getCamera();

        // --- ADD THESE TWO LINES BACK ---
        const glm::mat4& getViewMatrix() const;
        const glm::mat4& getProjectionMatrix() const;
        // --- END ADD ---

        std::shared_ptr<Mesh> getTriangleMesh() const;

        asset::ModelLoader* getModelLoader() const;

    private:
        std::unique_ptr<gl::Shader> shader_; // <-- Use gl::Shader
        std::unique_ptr<Camera> camera_;
        std::shared_ptr<Mesh> m_TriangleMesh;

        std::unique_ptr<asset::ModelLoader> m_ModelLoader;
    };

} // namespace Umgebung::renderer