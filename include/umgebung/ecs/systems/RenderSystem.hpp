/**
 * @file RenderSystem.hpp
 * @brief Contains the RenderSystem class.
 */
#pragma once

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
         * @brief Called every frame to update the system.
         * 
         * @param scene The scene to render.
         */
        void onUpdate(scene::Scene& scene);

    private:
        renderer::Renderer* renderer_; ///< The renderer to use.
    };

}