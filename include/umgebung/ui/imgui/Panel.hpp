#pragma once

#include <string>
#include <imgui.h> // <-- Include ImGui header for the flags type

namespace Umgebung::ui::imgui {

    class Panel {
    public:
        // Update the constructor to accept flags, with a default of none.
        explicit Panel(std::string name, ImGuiWindowFlags flags = ImGuiWindowFlags_None);
        virtual ~Panel() = default;

        virtual void onUIRender() = 0;

    protected:
        std::string name_;

        // Add a member to store the flags for this panel
        ImGuiWindowFlags flags_;
    };

} // namespace Umgebung::ui::imgui