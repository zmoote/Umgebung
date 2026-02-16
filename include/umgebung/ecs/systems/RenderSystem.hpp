/**
 * @file RenderSystem.hpp
 * @brief Contains the RenderSystem class.
 */
#pragma once

#include "umgebung/renderer/Camera.hpp"
#include "umgebung/ecs/components/ScaleComponent.hpp"
#include <entt/entt.hpp>

namespace Umgebung::renderer { class Renderer; }
namespace Umgebung::scene { class Scene; }

namespace Umgebung::ecs::systems {

    /**
     * @brief A system that renders entities with a Transform and Renderable component.
     */
    class RenderSystem {
    public:
        /**
         * @brief Construct a new Render System object.
         * 
         * @param renderer The renderer to use.
         */
        explicit RenderSystem(renderer::Renderer* renderer);

        /**
         * @brief Updates the system.
         * 
         * @param scene The scene to render.
         * @param camera The camera to render from.
         * @param selectedEntity The currently selected entity (optional).
         * @param observerScale The current scale of the observer (optional).
         */
        void onUpdate(scene::Scene& scene, const renderer::Camera& camera, entt::entity selectedEntity = entt::null, components::ScaleType observerScale = components::ScaleType::Human);

    private:
        renderer::Renderer* renderer_; ///< The renderer to use.
    };

}