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

                    // --- Add the log rendering logic ---

                    // Add a "Clear" button
                    if (ImGui::Button("Clear")) {
                        util::Logger::instance().clearPanelSinkBuffer();
                    }
                    ImGui::Separator();

                    // Create a scrolling region for the log messages
                    ImGui::BeginChild("ScrollingRegion", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

                    // Get the log buffer from our logger
                    const auto& logBuffer = util::Logger::instance().getPanelSinkBuffer();

                    // Display each message
                    for (const auto& msg : logBuffer) {
                        ImGui::TextUnformatted(msg.c_str());
                    }

                    // Auto-scroll to the bottom if new messages are added
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