#pragma once

#include <string>
#include <imgui.h>

namespace Umgebung {
    namespace ui {
        namespace imgui {
            /**
             * @class Panel
             * @brief An abstract base class for all ImGui panels in the application.
             *
             * This class provides a common interface for panels to be rendered
             * and managed by the main application.
             */
            class Panel {
            public:
                /**
                 * @brief Constructor.
                 * @param title The default title of the panel window.
                 */
                Panel(const std::string& title)
                    : m_title(title), m_isOpen(true), m_flags(0) {
                }

                /**
                 * @brief Virtual destructor.
                 * Important for base classes to ensure proper cleanup of derived types.
                 */
                virtual ~Panel() = default;

                /**
                 * @brief The main render function for the panel.
                 * This is a pure virtual function, meaning derived classes MUST implement it.
                 */
                virtual void render() = 0;

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

                /**
                 * @brief Gets the title of the panel.
                 */
                const std::string& getTitle() const { return m_title; }

            protected:
                std::string m_title;
                bool m_isOpen;
                ImGuiWindowFlags m_flags;
            };
        }
    }
}