#pragma once

#include <entt/entt.hpp>

namespace Umgebung::scene {

    class Scene {
    public:
        Scene();
        ~Scene();

        entt::entity createEntity();
        void destroyEntity(entt::entity entity);

        void onUpdate(float ts);

        entt::registry& getRegistry() { return registry_; }

        void setSelectedEntity(entt::entity entity) { m_SelectedEntity = entity; }
        entt::entity getSelectedEntity() const { return m_SelectedEntity; }

    private:
        entt::registry registry_;

        entt::entity m_SelectedEntity{ entt::null };
    };

}