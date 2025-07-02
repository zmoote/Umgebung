#include <umgebung/core/Application.hpp>
#include <umgebung/utils/Logger.hpp>

int main() {
    umgebung::utils::Logger::init();
    umgebung::core::Application app(1280, 720, "Umgebung");
    app.run();
    return 0;
}