#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung::ui::imgui {

    // The constructor now initializes the name and the flags
    Panel::Panel(std::string name, bool m_isOpen, ImGuiWindowFlags flags)
        : name_(std::move(name)), m_isOpen(m_isOpen), flags_(flags) {
    }

} // namespace Umgebung::ui::imgui