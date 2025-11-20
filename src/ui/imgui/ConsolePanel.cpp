/**
 * @file ConsolePanel.cpp
 * @brief Implements the ConsolePanel class.
 */
#include "umgebung/ui/imgui/ConsolePanel.hpp"
#include "umgebung/util/Logger.hpp"
#include <imgui.h>

namespace Umgebung {
    namespace ui {
        namespace imgui {

            ConsolePanel::ConsolePanel()
                : Panel("Console")
            {

            }

            void ConsolePanel::onUIRender() {

                if (!m_isOpen) {
                    return;
                }
                
                if (ImGui::Begin(name_.c_str(), &m_isOpen, flags_)) {

                    if (ImGui::Button("Clear")) {
                        util::Logger::instance().clearPanelSinkBuffer();
                    }
                    ImGui::Separator();

                    ImGui::BeginChild("ScrollingRegion", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

                    const auto& logBuffer = util::Logger::instance().getPanelSinkBuffer();

                    for (const auto& msg : logBuffer) {
                        ImGui::TextUnformatted(msg.c_str());
                    }

                    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
                        ImGui::SetScrollHereY(1.0f);
                    }

                    ImGui::EndChild();

                }
                ImGui::End();
            }

        }
    }
}