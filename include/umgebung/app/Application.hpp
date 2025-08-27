#pragma once

#include <memory>
#include <vector>
#include "umgebung/ui/Window.hpp"
#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung
{
    namespace app
    {
        /**
         * @class Application
         * @brief Main application class that orchestrates the entire program.
         *
         * This class owns the main window and runs the main application loop.
         */
        class Application
        {
        public:
            /**
             * @brief Default constructor.
             */
            Application();

            /**
             * @brief Default destructor.
             */
            ~Application();

            /**
             * @brief Initializes the application and its subsystems.
             * @return 0 on success, non-zero on failure.
             */
            int init();

            /**
             * @brief Runs the main application loop.
             */
            void run();

        private:
            std::unique_ptr<ui::Window> m_window;
            bool m_isRunning = false;

            // A list to hold all of our UI panels
            std::vector<std::unique_ptr<ui::imgui::Panel>> m_panels;
        };
    } // namespace app
} // namespace Umgebung