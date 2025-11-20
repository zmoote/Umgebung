#pragma once

#include "umgebung/renderer/DebugRenderer.hpp"
#include <entt/entt.hpp>

namespace Umgebung::ecs::systems
{

    class DebugRenderSystem
    {
    public:
        DebugRenderSystem(renderer::DebugRenderer* debugRenderer);
        void onUpdate(entt::registry& registry);

        void setEnabled(bool enabled) { enabled_ = enabled; }
        bool isEnabled() const { return enabled_; }

    private:
        renderer::DebugRenderer* debugRenderer_;
        bool enabled_ = true;
    };

} // namespace Umgebung::ecs::systems
