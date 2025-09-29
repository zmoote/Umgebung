#pragma once

#include <string>
#include <imgui.h>

namespace Umgebung::ui::imgui {

    class Panel {
    public:
        explicit Panel(std::string name, bool m_isOpen = true, ImGuiWindowFlags flags = ImGuiWindowFlags_None);
        virtual ~Panel() = default;

        virtual void onUIRender() = 0;

        bool isOpen() const { return m_isOpen; }

        void open() { m_isOpen = true; }

        void close() { m_isOpen = false; }


    protected:
        std::string name_;

        ImGuiWindowFlags flags_;

        bool m_isOpen;
    };

}