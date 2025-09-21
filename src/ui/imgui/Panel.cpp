#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung::ui::imgui {

    // The constructor now initializes the name and the flags
    Panel::Panel(std::string name, ImGuiWindowFlags flags)
        : name_(std::move(name)), flags_(flags) {
    }

} // namespace Umgebung::ui::imgui