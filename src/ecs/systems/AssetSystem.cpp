/**
 * @file AssetSystem.cpp
 * @brief Implements the AssetSystem class.
 */
#include "umgebung/ecs/systems/AssetSystem.hpp"
#include "umgebung/asset/ModelLoader.hpp"
#include "umgebung/scene/Scene.hpp"
#include "umgebung/ecs/components/Renderable.hpp"

namespace Umgebung::ecs::systems {

    AssetSystem::AssetSystem(asset::ModelLoader* modelLoader)
        : modelLoader_(modelLoader) {}

    void AssetSystem::onUpdate(scene::Scene& scene) {
        auto& registry = scene.getRegistry();
        auto view = registry.view<components::Renderable>();

        for (auto [entity, renderable] : view.each()) {
            if (renderable.meshTag != renderable.loadedMeshTag) {
                renderable.mesh = modelLoader_->loadMesh(renderable.meshTag);
                renderable.loadedMeshTag = renderable.meshTag;
            }
        }
    }

} // namespace Umgebung::ecs::systems
