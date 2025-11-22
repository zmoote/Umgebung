/**
 * @file FilePickerPanel.hpp
 * @brief Defines the FilePickerPanel class.
 */
#pragma once

#include "umgebung/ui/imgui/Panel.hpp"
#include <functional>
#include <string>
#include <vector>
#include <filesystem>

namespace Umgebung::ui::imgui {

class FilePickerPanel : public Panel {
public:
    using FileSelectedCallback = std::function<void(const std::filesystem::path&)>;

    FilePickerPanel();

    void open(const std::string& title, const std::string& buttonLabel, FileSelectedCallback callback, const std::vector<std::string>& extensions);
    void onUIRender() override;

private:
    std::string title_;
    std::string buttonLabel_;
    FileSelectedCallback callback_;
    std::vector<std::string> extensions_;
    std::filesystem::path currentPath_;
    char inputBuffer_[256] = {};
};

} // namespace Umgebung::ui::imgui

