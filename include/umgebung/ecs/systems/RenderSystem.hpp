#pragma once

// Forward-declarations to keep header dependencies clean
namespace Umgebung::renderer { class Renderer; }
namespace Umgebung::scene { class Scene; }

namespace Umgebung::ecs::systems {

    /**
     * @class RenderSystem
     * @brief Processes entities with renderable components and issues draw calls.
     *
     * This system queries the scene for entities that have both a TransformComponent
     * and a RenderableComponent, then uses the main Renderer to draw them.
     */
    class RenderSystem {
    public:
        /**
         * @brief Constructs the RenderSystem.
         * @param renderer A pointer to the main Renderer instance.
         */
        explicit RenderSystem(renderer::Renderer* renderer);

        /**
         * @brief Renders one frame of the scene.
         * @param scene The scene to render.
         */
        void onUpdate(scene::Scene& scene);

    private:
        renderer::Renderer* renderer_;
    };

} // namespace Umgebung::ecs::systems