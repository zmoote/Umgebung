/**
 * @file Multiverse.hpp
 * @brief Top-level domain container managing the collection of `.umgebung`
 *        `UniverseScene` instances within the Umgebung Multiverse.
 *
 * @details
 * Reflects Umgebung's nested cosmological architecture (see `GEMINI.md`):
 * a single `Multiverse` acts as the overarching structural container holding
 * one or more independent `UniverseScene` bubbles, each persisted in its own
 * `.umgebung` file with its own `PhysicsProfile` (`Mainstream`, `Unified`, or
 * `Custom`).
 *
 * Both Headless HPC batch mode (`--scene`) and the Desktop Interactive GUI
 * (**Multiverse Hierarchy** panel) query this container directly.
 */

#pragma once

#include "umgebung/scene/UniverseScene.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace umgebung::scene {

    /**
     * @class Multiverse
     * @brief Owns and indexes loaded `UniverseScene` instances and tracks the
     *        currently active universe selected for simulation.
     */
    class Multiverse final {
    public:
        /** @brief Constructs an empty `Multiverse` container. */
        Multiverse() = default;

        /**
         * @brief Loads a single `.umgebung` file from disk, appends it to this
         *        multiverse, and sets it as the active universe if it is the first.
         *
         * @param      filePath Path to the `.umgebung` file.
         * @param[out] outError Populated with diagnostics if loading or validation fails.
         * @return `true` on success; `false` on I/O, syntax, or validation error.
         */
        [[nodiscard]] bool LoadUniverseFromFile(
            const std::filesystem::path& filePath,
            std::string& outError
        );

        /**
         * @brief Scans a directory (non-recursively) in deterministic sorted
         *        filename order and loads every `*.umgebung` file found.
         *
         * @details
         * Individual malformed files are logged as warnings via `spdlog` without
         * aborting discovery of other valid universe scenes in the directory.
         *
         * @param directoryPath Directory containing `.umgebung` universe files.
         * @return Total number of valid universe scenes loaded from `directoryPath`.
         */
        [[nodiscard]] std::size_t LoadDirectory(const std::filesystem::path& directoryPath);

        /**
         * @brief Inserts a programmatically constructed `UniverseScene` into the
         *        multiverse after validating its invariants.
         *
         * @param      scene    Scene to add.
         * @param[out] outError Diagnostic message if `scene.Validate()` fails.
         * @return `true` if added; `false` if invalid.
         */
        [[nodiscard]] bool AddUniverse(UniverseScene scene, std::string& outError);

        /** @brief Removes all loaded universes and resets active selection. */
        void Clear() noexcept;

        /** @brief Returns the number of loaded universes in this multiverse. */
        [[nodiscard]] std::size_t GetUniverseCount() const noexcept { return m_universes.size(); }

        /** @brief Returns `true` when zero universes are loaded. */
        [[nodiscard]] bool IsEmpty() const noexcept { return m_universes.empty(); }

        /** @brief Returns read-only access to all loaded `UniverseScene` objects. */
        [[nodiscard]] const std::vector<UniverseScene>& GetUniverses() const noexcept { return m_universes; }

        /** @brief Returns the zero-based index of the currently active universe. */
        [[nodiscard]] std::size_t GetActiveUniverseIndex() const noexcept { return m_activeIndex; }

        /**
         * @brief Changes the active universe selection.
         *
         * @param index Zero-based index in `[0, GetUniverseCount())`.
         * @return `true` if `index` was valid; `false` if out of range.
         */
        [[nodiscard]] bool SetActiveUniverseIndex(std::size_t index) noexcept;

        /** @brief Returns a pointer to the currently active `UniverseScene`, or `nullptr` if empty. */
        [[nodiscard]] const UniverseScene* GetActiveUniverse() const noexcept;

        /** @brief Mutable overload returning a pointer to the active `UniverseScene`, or `nullptr` if empty. */
        [[nodiscard]] UniverseScene* GetActiveUniverse() noexcept;

    private:
        /** @brief Ordered list of universes owned by this multiverse. */
        std::vector<UniverseScene> m_universes;

        /** @brief Index of the currently active universe selected for simulation/UI inspection. */
        std::size_t m_activeIndex{0U};
    };

} // namespace umgebung::scene
