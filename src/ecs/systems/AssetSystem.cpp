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

    bool AssetSystem::onUpdate(scene::Scene& scene) {
        auto& registry = scene.getRegistry();
        auto view = registry.view<components::Renderable>();

        bool anyLoaded = false;
        // Track how many we load in one frame to avoid hitches
        int loadCount = 0;
        const int MAX_LOADS_PER_FRAME = 100;

        for (auto [entity, renderable] : view.each()) {
            if (renderable.meshTag.empty()) continue;

            if (renderable.meshTag != renderable.loadedMeshTag) {
                // Optimization: Don't log for every single entity if there are thousands
                if (view.size() < 100) {
                    UMGEBUNG_LOG_INFO("AssetSystem: Loading mesh for entity {}: '{}' -> '{}'", 
                        static_cast<uint32_t>(entity), renderable.loadedMeshTag, renderable.meshTag);
                }
                
                renderable.mesh = modelLoader_->loadMesh(renderable.meshTag);
                renderable.loadedMeshTag = renderable.meshTag;
                
                loadCount++;
                anyLoaded = true;
                if (loadCount >= MAX_LOADS_PER_FRAME && view.size() > 100) {
                    break; // Spread loading over multiple frames for large scenes
                }
            }
        }
        return anyLoaded;
    }

} // namespace Umgebung::ecs::systems
