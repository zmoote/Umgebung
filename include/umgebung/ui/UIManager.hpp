/**
 * @file UIManager.hpp
 * @brief Top-level Dear ImGui workstation manager orchestrating the legacy
 *        Umgebung 4-region docked layout, Main Menu Bar, `ImGuiConsoleSink`,
 *        modal `FilePickerPanel`, and all modular panels.
 *
 * @details
 * Recreates and adapts the exact workstation UI architecture shared across all
 * branches of `Umgebung-legacy` (`opengl-backend`, `vulkan-backend`, `main`,
 * `dev`):
 *   - Full-viewport `MainDockSpace` (`ImGuiDockNodeFlags_PassthruCentralNode`),
 *   - Deterministic first-run (and resettable) 4-region `DockBuilder` layout:
 *       - **Left (`20%`):** `"Hierarchy"` (`HierarchyPanel`)
 *       - **Right (`25%`):** `"Properties"` (`PropertiesPanel`)
 *       - **Bottom (`25%` under center):** `"Console"` (`ConsolePanel`)
 *       - **Center:** `"Viewport"` (`ViewportPanel`)
 *   - Floating non-docking utility panels opened on demand from the menu bar:
 *       - `Tools -> Statistics` (`StatisticsPanel`)
 *       - `Help -> About Umgebung` (`AboutPanel`)
 *       - `File -> Open Scene... / Save As...` (`FilePickerPanel` modal)
 *   - Main Menu Bar matching legacy Umgebung (`File`, `Simulation`, `Modules`,
 *     `Tools`, `View`, `Help`) plus a compact right-aligned live status strip.
 *
 * @warning Compiled strictly when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#pragma once

#if !defined(UMGEBUNG_HAS_GUI) || (UMGEBUNG_HAS_GUI == 0)
    #error "umgebung/ui/UIManager.hpp is exclusive to GUI builds (UMGEBUNG_BUILD_GUI=ON)."
#endif

#include "umgebung/ui/ImGuiConsoleSink.hpp"
#include "umgebung/ui/Panel.hpp"
#include "umgebung/ui/UIContext.hpp"
#include "umgebung/ui/panels/FilePickerPanel.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace umgebung::ui {

    /**
     * @class UIManager
     * @brief Owns and renders all modular Dear ImGui panels, the default
     *        4-pane docking layout, the main menu bar, and the `spdlog` console sink.
     */
    class UIManager final {
    public:
        /** @brief Constructs an uninitialized `UIManager`; call `Initialize()` after ImGui context creation. */
        UIManager();

        /** @brief Destructor; guarantees clean detachment of `ImGuiConsoleSink` from `spdlog`. */
        ~UIManager();

        UIManager(const UIManager&) = delete;
        UIManager& operator=(const UIManager&) = delete;
        UIManager(UIManager&&) = delete;
        UIManager& operator=(UIManager&&) = delete;

        /**
         * @brief Attaches `ImGuiConsoleSink` to `spdlog::default_logger()` and
         *        instantiates all modular workstation and utility panels.
         *
         * @post `IsInitialized() == true` and all default panels are registered.
         */
        void Initialize();

        /**
         * @brief Detaches `ImGuiConsoleSink` from `spdlog` and destroys all panels.
         *
         * @details Idempotent; safe to call during `Application::Shutdown()` and `~UIManager()`.
         */
        void Shutdown() noexcept;

        /**
         * @brief Renders the full-viewport dockspace, main menu bar, all active
         *        modular panels, and the modal file picker for the current frame.
         *
         * @param ctx Live engine state and action callbacks for this frame.
         *
         * @pre `ImGui::NewFrame()` has already been called for the current frame.
         */
        void Render(UIContext& ctx);

        /**
         * @brief Requests a rebuild of the canonical legacy 4-region `DockBuilder`
         *        layout (`Hierarchy` Left 20%, `Properties` Right 25%, `Console`
         *        Bottom 25%, `Viewport` Center) on the next frame.
         */
        void RequestResetLayout() noexcept { m_forceResetLayout = true; }

        /**
         * @brief Opens the modal `FilePickerPanel` for browsing/opening/saving files.
         *
         * @param title       Modal dialog window title.
         * @param buttonLabel Primary confirmation button text.
         * @param callback    Callback receiving the selected `std::filesystem::path`.
         * @param extensions  Allowed filename suffix filters.
         * @param startPath   Optional initial directory to browse.
         * @param defaultName Optional initial filename string.
         */
        void OpenFilePicker(
            const std::string& title,
            const std::string& buttonLabel,
            FilePickerPanel::FileSelectedCallback callback,
            const std::vector<std::string>& extensions,
            const std::filesystem::path& startPath = {},
            const std::string& defaultName = {}
        );

        /**
         * @brief Looks up a registered panel by concrete subclass type `T`.
         *
         * @tparam T Subclass of `umgebung::ui::Panel`.
         * @return Pointer to the matching panel instance, or `nullptr` if absent.
         */
        template<typename T>
        [[nodiscard]] T* GetPanel() noexcept {
            for (const auto& panel : m_panels) {
                if (auto* typed = dynamic_cast<T*>(panel.get())) {
                    return typed;
                }
            }
            return nullptr;
        }

        /** @brief Returns `true` if `Initialize()` has been called and not yet shut down. */
        [[nodiscard]] bool IsInitialized() const noexcept { return m_initialized; }

        /** @brief Returns the shared `ImGuiConsoleSink` attached to `spdlog`. */
        [[nodiscard]] std::shared_ptr<ImGuiConsoleSink> GetConsoleSink() const noexcept {
            return m_consoleSink;
        }

    private:
        /** @brief Establishes `MainDockSpace` and builds the default 4-region layout if needed. */
        void SetupDockspace();

        /** @brief Renders the legacy-adapted Main Menu Bar (`File`, `Simulation`, `Modules`, `Tools`, `View`, `Help`). */
        void RenderMainMenuBar(UIContext& ctx);

        /** @brief Resolves a sensible default directory for Open/Save file-picker dialogs (`scenes/`). */
        [[nodiscard]] std::filesystem::path ResolveDefaultScenesDirectory(const UIContext& ctx) const;

        bool m_initialized{false};
        bool m_layoutInitialized{false};
        bool m_forceResetLayout{false};

        std::shared_ptr<ImGuiConsoleSink> m_consoleSink;
        std::vector<std::unique_ptr<Panel>> m_panels;
        std::unique_ptr<FilePickerPanel> m_filePickerPanel;
    };

} // namespace umgebung::ui
