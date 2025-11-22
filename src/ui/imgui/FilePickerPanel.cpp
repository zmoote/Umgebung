#include "umgebung/ui/imgui/FilePickerPanel.hpp"
#include <imgui.h>

namespace Umgebung::ui::imgui {

FilePickerPanel::FilePickerPanel() : Panel("File Picker", false) {
    currentPath_ = std::filesystem::current_path();
}

void FilePickerPanel::open(const std::string& title, const std::string& buttonLabel, FileSelectedCallback callback, const std::vector<std::string>& extensions) {
    title_ = title;
    buttonLabel_ = buttonLabel;
    callback_ = callback;
    extensions_ = extensions;
    m_isOpen = true;
    strcpy_s(inputBuffer_, "");
}

void FilePickerPanel::onUIRender() {
    if (!m_isOpen) {
        return;
    }

    ImGui::OpenPopup(title_.c_str());

    if (ImGui::BeginPopupModal(title_.c_str(), &m_isOpen)) {
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
                bool has_valid_extension = false;
                if (extensions_.empty()) {
                    has_valid_extension = true;
                } else {
                    for (const auto& ext : extensions_) {
                        if (path.extension() == ext) {
                            has_valid_extension = true;
                            break;
                        }
                    }
                }

                if (has_valid_extension) {
                    if (ImGui::Selectable(filename.c_str())) {
                        strcpy_s(inputBuffer_, filename.c_str());
                    }
                }
            }
        }

        ImGui::Separator();
        
        ImGui::InputText("Filename", inputBuffer_, sizeof(inputBuffer_));

        if (ImGui::Button(buttonLabel_.c_str())) {
            callback_(currentPath_ / inputBuffer_);
            m_isOpen = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Close")) {
            m_isOpen = false;
        }

        ImGui::EndPopup();
    }
}

} // namespace Umgebung::ui::imgui
