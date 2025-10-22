#pragma once

#include <string>

// Forward-declare the Scene class to avoid circular includes
namespace Umgebung::scene {
    class Scene;
}

namespace Umgebung::scene {

    class SceneSerializer {
    public:
        // Constructor takes a pointer to the scene it will operate on
        SceneSerializer(Scene* scene);

        void serialize(const std::string& filepath);
        bool deserialize(const std::string& filepath);

    private:
        Scene* m_Scene; // The scene to serialize/deserialize
    };

} // namespace Umgebung::scene