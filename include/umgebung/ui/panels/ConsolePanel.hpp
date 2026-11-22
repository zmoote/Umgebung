/**
 * @file ConsolePanel.hpp
 * @brief Docked bottom-pane Console panel streaming live `spdlog` engine logs.
 *
 * @details
 * Mirrors the bottom-docked `ConsolePanel` from every branch of `Umgebung-legacy`
 * (`25%` bottom dock beneath `Viewport`), featuring a `Clear` button, severity
 * level filter, auto-scroll toggle, horizontal scrollbar, and level-based color
 * highlighting fed thread-safely by `ImGuiConsoleSink`.
 *
 * @warning Compiled strictly when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#pragma once

#if !defined(UMGEBUNG_HAS_GUI) || (UMGEBUNG_HAS_GUI == 0)
    #error "umgebung/ui/panels/ConsolePanel.hpp is exclusive to GUI builds (UMGEBUNG_BUILD_GUI=ON)."
#endif

#include "umgebung/ui/ImGuiConsoleSink.hpp"
#include "umgebung/ui/Panel.hpp"

#include <memory>
#include <vector>

namespace umgebung::ui {

    /**
     * @class ConsolePanel
     * @brief Docked log console rendering formatted `spdlog` output captured by `ImGuiConsoleSink`.
     */
    class ConsolePanel final : public Panel {
    public:
        /**
         * @brief Constructs a `ConsolePanel` bound to a shared `ImGuiConsoleSink`.
         *
         * @param consoleSink Shared sink attached to `spdlog::default_logger()`.
         */
        explicit ConsolePanel(std::shared_ptr<ImGuiConsoleSink> consoleSink);

        /**
         * @brief Renders the docked `"Console"` window and its scrollable log buffer.
         *
         * @param ctx Shared UI context (unused directly, required by `Panel` interface).
         */
        void OnUIRender(UIContext& ctx) override;

    private:
        /** @brief Thread-safe `spdlog` sink providing buffered log lines. */
        std::shared_ptr<ImGuiConsoleSink> m_consoleSink;

        /** @brief Reusable frame-local snapshot buffer avoiding per-frame allocations. */
        std::vector<ConsoleLogEntry> m_snapshotBuffer;

        /** @brief When `true`, automatically pins the scrollbar to the newest log line. */
        bool m_autoScroll{true};

        /** @brief Minimum `spdlog` severity level shown (`0`=trace .. `4`=error). */
        int m_minSeverityFilter{0};
    };

} // namespace umgebung::ui
