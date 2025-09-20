#pragma once

#include <string>

namespace Umgebung::ui::imgui {

    class Panel {
    public:
        // Constructor to set the panel's name
        explicit Panel(std::string name) : name_(std::move(name)) {}
        virtual ~Panel() = default;

        // This MUST be a virtual function for overriding to work.
        // Let's also make it a pure virtual function to ensure all panels must implement it.
        virtual void onUIRender() = 0;

    protected:
        // Add the missing name member so derived classes can access it
        std::string name_;
    };

} // namespace Umgebung::ui::imgui