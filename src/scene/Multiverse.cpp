/**
 * @file Multiverse.cpp
 * @brief Implementation of the `umgebung::scene::Multiverse` universe container
 *        and `.umgebung` directory discovery.
 */

#include "umgebung/scene/Multiverse.hpp"

#include <algorithm>
#include <utility>

#include <spdlog/spdlog.h>

namespace umgebung::scene {

    bool Multiverse::LoadUniverseFromFile(
        const std::filesystem::path& filePath,
        std::string& outError
    ) {
        UniverseScene scene{};
        if (!UniverseScene::LoadFromFile(filePath, scene, outError)) {
            return false;
        }

        m_universes.push_back(std::move(scene));
        if (m_universes.size() == 1U) {
            m_activeIndex = 0U;
        }

        const UniverseScene& loaded = m_universes.back();
        spdlog::info(
            "Multiverse: Loaded universe '{}' ({}) [profile={}, integrator={}, {} bodies] from '{}'.",
            loaded.name,
            loaded.universeId,
            PhysicsProfileTypeToString(loaded.physicsProfile.type),
            IntegratorTypeToString(loaded.physicsProfile.integrator),
            loaded.bodies.size(),
            filePath.string()
        );
        return true;
    }

    std::size_t Multiverse::LoadDirectory(const std::filesystem::path& directoryPath) {
        std::error_code ec;
        if (!std::filesystem::exists(directoryPath, ec) ||
            !std::filesystem::is_directory(directoryPath, ec)) {
            return 0U;
        }

        // Collect all `*.umgebung` file paths first (skipping `*.multiverse.umgebung`
        // manifests, which belong to `FlowerOfLifeMultiverse`) and sort
        // lexicographically so ordering is 100% deterministic across filesystems.
        std::vector<std::filesystem::path> sceneFiles;
        for (const auto& entry : std::filesystem::directory_iterator(directoryPath, ec)) {
            if (!entry.is_regular_file(ec)) {
                continue;
            }
            const std::string filename = entry.path().filename().string();
            if (entry.path().extension() == ".umgebung" &&
                filename.find(".multiverse.umgebung") == std::string::npos) {
                sceneFiles.push_back(entry.path());
            }
        }
        std::sort(sceneFiles.begin(), sceneFiles.end());

        std::size_t loadedCount = 0U;
        for (const auto& path : sceneFiles) {
            std::string errorMsg;
            if (LoadUniverseFromFile(path, errorMsg)) {
                ++loadedCount;
            } else {
                spdlog::warn("Multiverse: Skipping invalid scene '{}': {}", path.string(), errorMsg);
            }
        }

        return loadedCount;
    }

    bool Multiverse::AddUniverse(UniverseScene scene, std::string& outError) {
        if (!scene.Validate(outError)) {
            return false;
        }

        m_universes.push_back(std::move(scene));
        if (m_universes.size() == 1U) {
            m_activeIndex = 0U;
        }
        return true;
    }

    void Multiverse::Clear() noexcept {
        m_universes.clear();
        m_activeIndex = 0U;
    }

    bool Multiverse::SetActiveUniverseIndex(std::size_t index) noexcept {
        if (index >= m_universes.size()) {
            return false;
        }
        m_activeIndex = index;
        return true;
    }

    const UniverseScene* Multiverse::GetActiveUniverse() const noexcept {
        if (m_universes.empty() || m_activeIndex >= m_universes.size()) {
            return nullptr;
        }
        return &m_universes[m_activeIndex];
    }

    UniverseScene* Multiverse::GetActiveUniverse() noexcept {
        if (m_universes.empty() || m_activeIndex >= m_universes.size()) {
            return nullptr;
        }
        return &m_universes[m_activeIndex];
    }

} // namespace umgebung::scene
