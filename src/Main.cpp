#include "umgebung/app/Application.hpp"
#include "umgebung/util/Logger.hpp"
#include <memory>

int main(int argc, char** argv)
{
    // Initialize the logger as the very first step
    Umgebung::util::Logger::instance().init("Umgebung", Umgebung::util::Logger::Level::Trace);

    // Create and run the application using a smart pointer
    auto app = std::make_unique<Umgebung::app::Application>();
    if (app->init() == 0)
    {
        app->run();
    }

    // app->shutdown() will be called automatically by the destructor
    // when main finishes.

    return 0;
}