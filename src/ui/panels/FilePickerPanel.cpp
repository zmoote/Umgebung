/**
 * @file FilePickerPanel.cpp
 * @brief Cross-platform Dear ImGui modal file-browser implementation adapted
 *        from `Umgebung-legacy`'s `FilePickerPanel`.
 *
 * @note Compiled exclusively when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#include "umgebung/ui/panels/FilePickerPanel.hpp"

#include <algorithm>
#include <cstdio>
#include <system_error>
#include <utility>

#include <imgui.h>

namespace umgebung::ui {

    FilePickerPanel::FilePickerPanel()
        : Panel("File Picker", false, ImGuiWindowFlags_AlwaysAutoResize) {
        std::error_code ec;
        m_currentPath = std::filesystem::current_path(ec);
        if (ec) {
            m_currentPath = std::filesystem::path(".");
        }
        m_inputBuffer.fill('\0');
    }

    void FilePickerPanel::OpenModal(
        const std::string& title,
        const std::string& buttonLabel,
        FileSelectedCallback callback,
        const std::vector<std::string>& extensions,
        const std::filesystem::path& startPath,
        const std::string& defaultName
    ) {
        m_title = title.empty() ? std::string("File Picker") : title;
        m_buttonLabel = buttonLabel.empty() ? std::string("Select") : buttonLabel;
        m_callback = std::move(callback);
        m_extensions = extensions;
        m_isOpen = true;
        m_shouldOpenPopup = true;

        std::snprintf(m_inputBuffer.data(), m_inputBuffer.size(), "%s", defaultName.c_str());

        if (!startPath.empty()) {
            std::error_code ec;
            if (std::filesystem::is_directory(startPath, ec)) {
                m_currentPath = std::filesystem::weakly_canonical(startPath, ec);
                if (ec) {
                    m_currentPath = startPath;
                }
            }
        }
    }

    bool FilePickerPanel::MatchesAllowedExtensions(const std::string& filename) const {
        if (m_extensions.empty()) {
            return true;
        }
        // Match by trailing suffix so compound extensions such as
        // `.scenario.umgebung` and `.multiverse.umgebung` match cleanly.
        for (const auto& ext : m_extensions) {
            if (ext.empty()) {
                continue;
            }
            if (filename.size() >= ext.size() &&
                filename.compare(filename.size() - ext.size(), ext.size(), ext) == 0) {
                return true;
            }
        }
        return false;
    }

    void FilePickerPanel::OnUIRender(UIContext& /*ctx*/) {
        if (!m_isOpen) {
            return;
        }

        if (m_shouldOpenPopup) {
            ImGui::OpenPopup(m_title.c_str());
            m_shouldOpenPopup = false;
        }

        ImGui::SetNextWindowSize(ImVec2(560.0f, 420.0f), ImGuiCond_FirstUseEver);
        if (ImGui::BeginPopupModal(m_title.c_str(), &m_isOpen)) {
            // Navigation header: parent directory button + current absolute path.
            if (ImGui::Button("..")) {
                if (m_currentPath.has_parent_path() && m_currentPath.parent_path() != m_currentPath) {
                    m_currentPath = m_currentPath.parent_path();
                }
            }
            ImGui::SameLine();
            ImGui::TextDisabled("%s", m_currentPath.string().c_str());
            ImGui::Separator();

            // Enumerate directory safely (never throw on unreadable system directories).
            std::vector<std::filesystem::path> directories;
            std::vector<std::filesystem::path> files;
            std::error_code iterEc;
            for (const auto& entry : std::filesystem::directory_iterator(m_currentPath, iterEc)) {
                std::error_code entryEc;
                if (entry.is_directory(entryEc)) {
                    directories.push_back(entry.path());
                } else if (entry.is_regular_file(entryEc)) {
                    const std::string fn = entry.path().filename().string();
                    if (MatchesAllowedExtensions(fn)) {
                        files.push_back(entry.path());
                    }
                }
            }

            std::sort(directories.begin(), directories.end());
            std::sort(files.begin(), files.end());

            // Scrollable directory/file listing region.
            const float footerReserve = ImGui::GetFrameHeightWithSpacing() * 2.6f;
            if (ImGui::BeginChild("FileListRegion", ImVec2(0.0f, -footerReserve), true)) {
                for (const auto& dir : directories) {
                    const std::string dirLabel = "[DIR] " + dir.filename().string();
                    if (ImGui::Selectable(dirLabel.c_str(), false, ImGuiSelectableFlags_DontClosePopups)) {
                        m_currentPath = dir;
                    }
                }

                for (const auto& file : files) {
                    const std::string fn = file.filename().string();
                    const bool isSelected = (fn == m_inputBuffer.data());
                    if (ImGui::Selectable(
                            fn.c_str(),
                            isSelected,
                            ImGuiSelectableFlags_AllowDoubleClick | ImGuiSelectableFlags_DontClosePopups
                        )) {
                        std::snprintf(m_inputBuffer.data(), m_inputBuffer.size(), "%s", fn.c_str());
                        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && m_callback) {
                            m_callback(m_currentPath / fn);
                            m_isOpen = false;
                            ImGui::CloseCurrentPopup();
                        }
                    }
                }

                if (directories.empty() && files.empty()) {
                    ImGui::TextDisabled("(No matching files or folders in this directory)");
                }
            }
            ImGui::EndChild();

            ImGui::Separator();
            ImGui::SetNextItemWidth(-1.0f);
            ImGui::InputText("##FilenameInput", m_inputBuffer.data(), m_inputBuffer.size());

            const bool hasFilename = (m_inputBuffer[0] != '\0');
            ImGui::BeginDisabled(!hasFilename);
            if (ImGui::Button(m_buttonLabel.c_str(), ImVec2(110.0f, 0.0f))) {
                std::filesystem::path selectedPath = m_currentPath / std::string(m_inputBuffer.data());
                if (!selectedPath.has_extension() && !m_extensions.empty()) {
                    selectedPath += m_extensions.front();
                }
                if (m_callback) {
                    m_callback(selectedPath);
                }
                m_isOpen = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndDisabled();

            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(110.0f, 0.0f))) {
                m_isOpen = false;
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }
    }

} // namespace umgebung::ui
