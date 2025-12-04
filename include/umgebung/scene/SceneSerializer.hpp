/**
 * @file SceneSerializer.hpp
 * @brief Contains the SceneSerializer class.
 */
#pragma once

#include <filesystem>
#include <string>
#include "umgebung/renderer/Camera.hpp"

// Forward-declare the Scene class to avoid circular includes
namespace Umgebung::scene {
    class Scene;
}

namespace Umgebung::renderer {
    class Renderer;
}

namespace Umgebung::scene {

    /**
     * @brief A class for serializing and deserializing a scene.
     */
    class SceneSerializer {
    public:
        /**
         * @brief Construct a new Scene Serializer object.
         * 
         * @param scene The scene to serialize/deserialize.
         * @param renderer The renderer to use.
         */
        SceneSerializer(Scene* scene, renderer::Renderer* renderer);

        /**
         * @brief Serializes the scene to a file.
         * 
         * @param filepath The path to the file.
         * @param editorCamera Optional pointer to the editor camera to serialize.
         */
        void serialize(const std::filesystem::path& filepath, renderer::Camera* editorCamera = nullptr);

        /**
         * @brief Deserializes the scene from a file.
         * 
         * @param filepath The path to the file.
         * @param editorCamera Optional pointer to the editor camera to update.
         * @return true if the scene was deserialized successfully, false otherwise.
         */
        bool deserialize(const std::filesystem::path& filepath, renderer::Camera* editorCamera = nullptr);

    private:
        Scene* m_Scene; ///< The scene to serialize/deserialize.
        renderer::Renderer* m_Renderer; ///< The renderer to use.
    };

}