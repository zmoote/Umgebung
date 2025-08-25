#include <iostream>
#include "util/Logger.hpp"

int main(int argc, char** argv) {

    // Initialise once – you can pass a config file path if you want.
    Umgebung::util::Logger::Init("assets\\config\\config.json");

    auto log = Umgebung::util::Logger::Get();

    log->info("Application started (pid={})", getpid());
    log->debug("Debug details: x = {}, y = {}", 42, 3.14);

    return 0;
}