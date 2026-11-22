/**
 * @file Panel.hpp
 * @brief Abstract base class for every modular Dear ImGui window/panel in Umgebung.
 *
 * @details
 * Adapts the clean polymorphic `Panel` abstraction established across all
 * `Umgebung-legacy` branches (`opengl-backend`, `vulkan-backend`, `main`, `dev`).
 * Each docked workstation panel (`ViewportPanel`, `HierarchyPanel`,
 * `PropertiesPanel`, `ConsolePanel`) and floating utility panel
 * (`StatisticsPanel`, `AboutPanel`, `FilePickerPanel`) inherits from `Panel` so
 * `UIManager` can manage visibility, menu-bar toggles, focus tracking, and
 * per-frame dispatch uniformly.
 *
 * @warning Compiled strictly when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#pragma once

#if !defined(UMGEBUNG_HAS_GUI) || (UMGEBUNG_HAS_GUI == 0)
    #error "umgebung/ui/Panel.hpp is exclusive to GUI builds (UMGEBUNG_BUILD_GUI=ON)."
#endif

#include <string>

#include <imgui.h>

namespace umgebung::ui {

    struct UIContext;

    /**
     * @class Panel
     * @brief Polymorphic base class for all docked and floating Dear ImGui panels.
     *
     * @details
     * Encapsulates the window title (`m_name`, which also serves as the stable
     * `ImGui::DockBuilderDockWindow` identifier), open/closed visibility state
     * (`m_isOpen`), ImGui window flags (`m_flags`), and cached focus/hover flags
     * updated each frame by concrete subclasses.
     */
    class Panel {
    public:
        /**
         * @brief Constructs a `Panel` with its ImGui window title and initial state.
         *
         * @param name   Stable Dear ImGui window title (used by `DockBuilder` and `ImGui::Begin`).
         * @param isOpen Initial visibility state (`true` for default docked workstation panes,
         *               `false` for on-demand utility windows like `Statistics` and `About`).
         * @param flags  Bitmask of `ImGuiWindowFlags` passed to `ImGui::Begin`.
         */
        explicit Panel(
            std::string name,
            bool isOpen = true,
            ImGuiWindowFlags flags = ImGuiWindowFlags_None
        );

        /** @brief Virtual destructor for safe polymorphic ownership inside `UIManager`. */
        virtual ~Panel() = default;

        // Panels are managed by unique_ptr inside UIManager and not copied.
        Panel(const Panel&) = delete;
        Panel& operator=(const Panel&) = delete;
        Panel(Panel&&) = delete;
        Panel& operator=(Panel&&) = delete;

        /**
         * @brief Emits this panel's Dear ImGui widgets for the active frame.
         *
         * @param ctx Shared engine/simulation UI context providing read/write state
         *            and action callbacks into `core::Application`.
         */
        virtual void OnUIRender(UIContext& ctx) = 0;

        /** @brief Returns the stable ImGui window title of this panel. */
        [[nodiscard]] const std::string& GetName() const noexcept { return m_name; }

        /** @brief Returns `true` if this panel is currently visible/open. */
        [[nodiscard]] bool IsOpen() const noexcept { return m_isOpen; }

        /** @brief Returns a mutable pointer to the visibility flag for `ImGui::MenuItem` checkboxes. */
        [[nodiscard]] bool* GetOpenPtr() noexcept { return &m_isOpen; }

        /** @brief Makes this panel visible on the next UI frame. */
        void Open() noexcept { m_isOpen = true; }

        /** @brief Hides this panel on the next UI frame. */
        void Close() noexcept { m_isOpen = false; }

        /** @brief Toggles panel visibility between open and closed. */
        void Toggle() noexcept { m_isOpen = !m_isOpen; }

        /** @brief Returns `true` if this panel held Dear ImGui focus on the last rendered frame. */
        [[nodiscard]] bool IsFocused() const noexcept { return m_focused; }

        /** @brief Returns `true` if the mouse cursor hovered this panel on the last rendered frame. */
        [[nodiscard]] bool IsHovered() const noexcept { return m_hovered; }

    protected:
        /** @brief Stable ImGui window title and DockBuilder target name. */
        std::string m_name;

        /** @brief Window flags forwarded to `ImGui::Begin`. */
        ImGuiWindowFlags m_flags{ImGuiWindowFlags_None};

        /** @brief Controls whether this panel renders and binds to the close `[X]` button. */
        bool m_isOpen{true};

        /** @brief Cached `ImGui::IsWindowFocused()` result from the latest frame. */
        bool m_focused{false};

        /** @brief Cached `ImGui::IsWindowHovered()` result from the latest frame. */
        bool m_hovered{false};
    };

} // namespace umgebung::ui
