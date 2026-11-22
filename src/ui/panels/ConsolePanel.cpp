/**
 * @file ConsolePanel.cpp
 * @brief Implementation of the bottom-docked `"Console"` panel streaming live
 *        `spdlog` engine logs via `ImGuiConsoleSink`.
 *
 * @note Compiled exclusively when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#include "umgebung/ui/panels/ConsolePanel.hpp"

#include <utility>

#include <imgui.h>

namespace umgebung::ui {

    namespace {

        /**
         * @brief Maps a `spdlog` severity level to a readable dark-theme text color.
         */
        [[nodiscard]] ImVec4 SeverityToColor(spdlog::level::level_enum level) noexcept {
            switch (level) {
                case spdlog::level::trace:    return ImVec4(0.55f, 0.58f, 0.65f, 1.0f);
                case spdlog::level::debug:    return ImVec4(0.65f, 0.72f, 0.82f, 1.0f);
                case spdlog::level::info:     return ImVec4(0.86f, 0.91f, 0.98f, 1.0f);
                case spdlog::level::warn:     return ImVec4(1.00f, 0.82f, 0.36f, 1.0f);
                case spdlog::level::err:      return ImVec4(1.00f, 0.42f, 0.40f, 1.0f);
                case spdlog::level::critical: return ImVec4(1.00f, 0.24f, 0.45f, 1.0f);
                default:                      return ImVec4(0.86f, 0.91f, 0.98f, 1.0f);
            }
        }

    } // anonymous namespace

    ConsolePanel::ConsolePanel(std::shared_ptr<ImGuiConsoleSink> consoleSink)
        : Panel("Console", true), m_consoleSink(std::move(consoleSink)) {}

    void ConsolePanel::OnUIRender(UIContext& /*ctx*/) {
        if (!m_isOpen) {
            return;
        }

        if (ImGui::Begin(m_name.c_str(), &m_isOpen, m_flags)) {
            m_focused = ImGui::IsWindowFocused();
            m_hovered = ImGui::IsWindowHovered();

            // Compact legacy-style Console toolbar: Clear + Auto-Scroll + Severity Filter.
            if (ImGui::SmallButton("Clear")) {
                if (m_consoleSink != nullptr) {
                    m_consoleSink->Clear();
                }
            }
            ImGui::SameLine();
            ImGui::Checkbox("Auto-Scroll", &m_autoScroll);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(115.0f);
            static constexpr const char* kFilterItems[] = {"All (Trace+)", "Debug+", "Info+", "Warn+", "Error+"};
            ImGui::Combo("##LogLevelFilter", &m_minSeverityFilter, kFilterItems, IM_ARRAYSIZE(kFilterItems));

            if (m_consoleSink != nullptr) {
                m_consoleSink->CopyEntries(m_snapshotBuffer);
                ImGui::SameLine();
                ImGui::TextDisabled("(%zu lines)", m_snapshotBuffer.size());
            } else {
                m_snapshotBuffer.clear();
            }

            ImGui::Separator();

            if (ImGui::BeginChild("ScrollingRegion", ImVec2(0.0f, 0.0f), false, ImGuiWindowFlags_HorizontalScrollbar)) {
                for (const auto& entry : m_snapshotBuffer) {
                    if (static_cast<int>(entry.level) < m_minSeverityFilter) {
                        continue;
                    }
                    ImGui::PushStyleColor(ImGuiCol_Text, SeverityToColor(entry.level));
                    ImGui::TextUnformatted(entry.message.c_str());
                    ImGui::PopStyleColor();
                }

                if (m_autoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
                    ImGui::SetScrollHereY(1.0f);
                }
            }
            ImGui::EndChild();
        }
        ImGui::End();
    }

} // namespace umgebung::ui
