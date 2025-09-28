#pragma once

#include <string>
#include <imgui.h> // <-- Include ImGui header for the flags type

namespace Umgebung::ui::imgui {

    class Panel {
    public:
        // Update the constructor to accept flags, with a default of none.
        explicit Panel(std::string name, bool m_isOpen = true, ImGuiWindowFlags flags = ImGuiWindowFlags_None);
        virtual ~Panel() = default;

        virtual void onUIRender() = 0;

        /**
        * @brief Checks if the panel is currently set to be visible.
        */
        bool isOpen() const { return m_isOpen; }

        /**
         * @brief Opens the panel (sets it to be visible).
         */
        void open() { m_isOpen = true; }

        /**
         * @brief Closes the panel (sets it to be hidden).
         */
        void close() { m_isOpen = false; }


    protected:
        std::string name_;

        // Add a member to store the flags for this panel
        ImGuiWindowFlags flags_;

        bool m_isOpen;
    };

} // namespace Umgebung::ui::imgui