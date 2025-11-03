/**
 * @file SceneSerializer.hpp
 * @brief Contains the SceneSerializer class.
 */
#pragma once

#include <string>

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
         */
        void serialize(const std::string& filepath);

        /**
         * @brief Deserializes the scene from a file.
         * 
         * @param filepath The path to the file.
         * @return true if the scene was deserialized successfully, false otherwise.
         */
        bool deserialize(const std::string& filepath);

    private:
        Scene* m_Scene; ///< The scene to serialize/deserialize.
        renderer::Renderer* m_Renderer; ///< The renderer to use.
    };

}