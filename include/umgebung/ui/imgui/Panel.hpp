/**
 * @file Panel.hpp
 * @brief Contains the Panel class.
 */
#pragma once

#include <string>
#include <imgui.h>

namespace Umgebung::ui::imgui {

    /**
     * @brief An abstract base class for all UI panels.
     */
    class Panel {
    public:
        /**
         * @brief Construct a new Panel object.
         * 
         * @param name The name of the panel.
         * @param m_isOpen Whether the panel is open by default.
         * @param flags The ImGui window flags.
         */
        explicit Panel(std::string name, bool m_isOpen = true, ImGuiWindowFlags flags = ImGuiWindowFlags_None);

        /**
         * @brief Destroy the Panel object.
         */
        virtual ~Panel() = default;

        /**
         * @brief Renders the panel.
         */
        virtual void onUIRender() = 0;

        /**
         * @brief Returns whether the panel is open.
         * 
         * @return true if the panel is open, false otherwise.
         */
        bool isOpen() const { return m_isOpen; }

        /**
         * @brief Opens the panel.
         */
        void open() { m_isOpen = true; }

        /**
         * @brief Closes the panel.
         */
        void close() { m_isOpen = false; }

        /**
         * @brief Returns whether the panel is focused.
         * 
         * @return true if the panel is focused, false otherwise.
         */
        bool isFocused() const { return focused_; }
    protected:
        std::string name_;      ///< The name of the panel.

        ImGuiWindowFlags flags_; ///< The ImGui window flags.

        bool m_isOpen;          ///< Whether the panel is open.
    private:
        bool focused_ = false;  ///< Whether the panel is focused.
    };

}