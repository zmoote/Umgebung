#pragma once

#include <entt/entt.hpp>

namespace Umgebung::scene {

    class Scene {
    public:
        Scene();
        ~Scene();

        entt::entity createEntity();

        void onUpdate(float ts);

        entt::registry& getRegistry() { return registry_; }

    private:
        entt::registry registry_;
    };

}