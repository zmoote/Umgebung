#include "Core/Application.hpp"
#include "Core/Logger.hpp"

int main() {
    Umgebung::Logger::Init();

    Umgebung::Application app;
    app.Init();
    app.Run();
    app.Shutdown();

    return 0;
}