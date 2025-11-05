/**
 * @file FilePickerPanel.cpp
 * @brief Implements the FilePickerPanel class.
 */
#include "umgebung/ui/imgui/FilePickerPanel.hpp"
#include <imgui.h>

namespace Umgebung::ui::imgui {

FilePickerPanel::FilePickerPanel(const char* name, const std::filesystem::path& path, FileSelectedCallback callback)
    : Panel(name), currentPath_(path), callback_(callback) {}

void FilePickerPanel::onUIRender() {
    if (!m_isOpen) {
        return;
    }

    ImGui::OpenPopup(name_.c_str());

    if (ImGui::BeginPopupModal(name_.c_str(), &m_isOpen)) {
        if (ImGui::Button("..")) {
            if (currentPath_.has_parent_path()) {
                currentPath_ = currentPath_.parent_path();
            }
        }

        for (const auto& entry : std::filesystem::directory_iterator(currentPath_)) {
            const auto& path = entry.path();
            std::string filename = path.filename().string();

            if (entry.is_directory()) {
                if (ImGui::Button(filename.c_str())) {
                    currentPath_ /= filename;
                }
            } else {
                if (ImGui::Selectable(filename.c_str())) {
                    callback_(path);
                    m_isOpen = false;
                }
            }
        }

        ImGui::Separator();

        if (ImGui::Button("Close")) {
            m_isOpen = false;
        }

        ImGui::EndPopup();
    }
}

} // namespace Umgebung::ui::imgui
