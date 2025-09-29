#pragma once

namespace Umgebung::renderer { class Renderer; }
namespace Umgebung::scene { class Scene; }

namespace Umgebung::ecs::systems {

    class RenderSystem {
    public:
        explicit RenderSystem(renderer::Renderer* renderer);

        void onUpdate(scene::Scene& scene);

    private:
        renderer::Renderer* renderer_;
    };

}