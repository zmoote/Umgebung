/**
 * @file AssetSystem.hpp
 * @brief Contains the AssetSystem class.
 */
#pragma once

namespace Umgebung::asset { class ModelLoader; }
namespace Umgebung::scene { class Scene; }

namespace Umgebung::ecs::systems {

    /**
     * @brief A system that manages asset loading for components.
     */
    class AssetSystem {
    public:
        /**
         * @brief Construct a new Asset System object.
         *
         * @param modelLoader The model loader to use.
         */
        explicit AssetSystem(asset::ModelLoader* modelLoader);

        /**
         * @brief Called every frame to update the system.
         *
         * @param scene The scene to process.
         */
        void onUpdate(scene::Scene& scene);

    private:
        asset::ModelLoader* modelLoader_;
    };

} // namespace Umgebung::ecs::systems
