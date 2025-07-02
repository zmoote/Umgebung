#pragma once
#include <entt/entt.hpp>

namespace umgebung::scene {
    class Scene {
    public:
        Scene();
        entt::entity create_entity();
        void update(float delta_time);
        entt::registry& get_registry();

    private:
        entt::registry registry_;
    };
}