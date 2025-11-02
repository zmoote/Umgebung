#pragma once

#include <memory>
#include <string>

/**
 * @file Renderer.hpp
 * @brief Contains the Renderer class.
 */
#pragma once

#include <memory>
#include <string>

// Forward Declarations
namespace Umgebung::renderer {
    class Mesh;
    class Camera;
    namespace gl { class Shader; }
}
namespace Umgebung::asset {
    class ModelLoader;
}

#include "umgebung/renderer/Camera.hpp"
#include "umgebung/renderer/Mesh.hpp"

namespace Umgebung::renderer {

    /**
     * @brief The main rendering class.
     */
    class Renderer {
    public:
        /**
         * @brief Construct a new Renderer object.
         */
        Renderer();

        /**
         * @brief Destroy the Renderer object.
         */
        ~Renderer();

        /**
         * @brief Initializes the renderer.
         */
        void init();

        /**
         * @brief Shuts down the renderer.
         */
        void shutdown();

        /**
         * @brief Get the Shader object.
         * 
         * @return gl::Shader& 
         */
        gl::Shader& getShader();

        /**
         * @brief Get the Camera object.
         * 
         * @return Camera& 
         */
        Camera& getCamera();

        /**
         * @brief Get the View Matrix object.
         * 
         * @return const glm::mat4& 
         */
        const glm::mat4& getViewMatrix() const;

        /**
         * @brief Get the Projection Matrix object.
         * 
         * @return const glm::mat4& 
         */
        const glm::mat4& getProjectionMatrix() const;

        /**
         * @brief Get the Triangle Mesh object.
         * 
         * @return std::shared_ptr<Mesh> 
         */
        std::shared_ptr<Mesh> getTriangleMesh() const;

        /**
         * @brief Get the Model Loader object.
         * 
         * @return asset::ModelLoader* 
         */
        asset::ModelLoader* getModelLoader() const;

    private:
        std::unique_ptr<gl::Shader> shader_; ///< The shader.
        std::unique_ptr<Camera> camera_;     ///< The camera.
        std::shared_ptr<Mesh> m_TriangleMesh; ///< The triangle mesh.

        std::unique_ptr<asset::ModelLoader> m_ModelLoader; ///< The model loader.
    };

} // namespace Umgebung::renderer