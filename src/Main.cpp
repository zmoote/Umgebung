#include <iostream>
#include "umgebung/util/LogMacros.hpp"
#include "umgebung/app/Application.hpp"

int main(int argc, char** argv) {

    Umgebung::util::Logger::instance().init("Umgebung", Umgebung::util::Logger::Level::Trace);

    Umgebung::app::Application app;
    app.init();
    app.run();

    return 0;
}