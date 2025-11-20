/**
 * @file Panel.cpp
 * @brief Implements the Panel class.
 */
#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung::ui::imgui {
    Panel::Panel(std::string name, bool m_isOpen, ImGuiWindowFlags flags)
        : name_(std::move(name)), m_isOpen(m_isOpen), flags_(flags) {
    }
}