/**
 * @file Panel.cpp
 * @brief Implementation of `umgebung::ui::Panel` base constructor.
 *
 * @note Compiled exclusively when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#include "umgebung/ui/Panel.hpp"

#include <utility>

namespace umgebung::ui {

    Panel::Panel(std::string name, bool isOpen, ImGuiWindowFlags flags)
        : m_name(std::move(name)), m_flags(flags), m_isOpen(isOpen) {}

} // namespace umgebung::ui
