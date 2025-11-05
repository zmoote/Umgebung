/**
 * @file FilePickerPanel.hpp
 * @brief Defines the FilePickerPanel class.
 */
#pragma once

#include "umgebung/ui/imgui/Panel.hpp"
#include <functional>
#include <string>
#include <filesystem>

namespace Umgebung::ui::imgui {

class FilePickerPanel : public Panel {
public:
    using FileSelectedCallback = std::function<void(const std::filesystem::path&)>;

    FilePickerPanel(const char* name, const std::filesystem::path& path, FileSelectedCallback callback);

    void onUIRender() override;

private:
    std::filesystem::path currentPath_;
    FileSelectedCallback callback_;
};

} // namespace Umgebung::ui::imgui
