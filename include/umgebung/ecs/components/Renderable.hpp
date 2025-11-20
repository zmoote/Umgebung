/**
 * @file Renderable.hpp
 * @brief Contains the Renderable component.
 */
#pragma once

#include "umgebung/renderer/Mesh.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>

namespace Umgebung::ecs::components {

    /**
     * @brief A component that makes an entity renderable.
     * 
     * This component holds the mesh, color, and a tag for the mesh.
     */
    struct Renderable {
        std::shared_ptr<renderer::Mesh> mesh; ///< The mesh to be rendered.
        glm::vec4 color{ 1.0f, 1.0f, 1.0f, 1.0f }; ///< The color of the mesh.

        std::string meshTag; ///< A tag for the mesh, used for serialization.
        std::string loadedMeshTag; ///< The tag of the mesh that is currently loaded.

        /**
         * @brief Construct a new Renderable object.
         * 
         * @param m The mesh.
         * @param c The color.
         * @param tag The mesh tag.
         */
        Renderable(std::shared_ptr<renderer::Mesh> m,
            glm::vec4 c = { 1.0f, 1.0f, 1.0f, 1.0f },
            std::string tag = "")
            : mesh(m), color(c), meshTag(tag) {
        }

        /**
         * @brief Default constructor for serialization.
         */
        Renderable() = default;
    };

    /**
     * @brief Teach JSON how to save/load this component.
     * 
     * We will NOT save the 'mesh' pointer. We only save the color and tag.
     */
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Renderable, color, meshTag)

}