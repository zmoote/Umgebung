#include "umgebung/app/Application.hpp"
#include "umgebung/util/Logger.hpp"
#include <memory>

int main(int argc, char** argv)
{
    Umgebung::util::Logger::instance().init("Umgebung", Umgebung::util::Logger::Level::Trace);

    auto app = std::make_unique<Umgebung::app::Application>();
    if (app->init() == 0)
    {
        app->run();
    }

    return 0;
}