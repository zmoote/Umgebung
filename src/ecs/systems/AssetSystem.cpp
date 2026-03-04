/**
 * @file AssetSystem.cpp
 * @brief Implements the AssetSystem class.
 */
#include "umgebung/ecs/systems/AssetSystem.hpp"
#include "umgebung/asset/ModelLoader.hpp"
#include "umgebung/scene/Scene.hpp"
#include "umgebung/ecs/components/Renderable.hpp"
#include "umgebung/util/LogMacros.hpp"

namespace Umgebung::ecs::systems {

    AssetSystem::AssetSystem(asset::ModelLoader* modelLoader)
        : modelLoader_(modelLoader) {}

    void AssetSystem::onUpdate(scene::Scene& scene) {
        auto& registry = scene.getRegistry();
        auto view = registry.view<components::Renderable>();

        for (auto [entity, renderable] : view.each()) {
            if (renderable.meshTag != renderable.loadedMeshTag) {
                UMGEBUNG_LOG_INFO("AssetSystem: Loading mesh for entity {}: '{}' -> '{}'", 
                    static_cast<uint32_t>(entity), renderable.loadedMeshTag, renderable.meshTag);
                
                renderable.mesh = modelLoader_->loadMesh(renderable.meshTag);
                renderable.loadedMeshTag = renderable.meshTag;
                
                if (renderable.mesh) {
                    UMGEBUNG_LOG_TRACE("AssetSystem: Successfully loaded mesh '{}'", renderable.meshTag);
                } else {
                    UMGEBUNG_LOG_ERROR("AssetSystem: Failed to load mesh '{}'", renderable.meshTag);
                }
            }
        }
    }

} // namespace Umgebung::ecs::systems
