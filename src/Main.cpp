#include <iostream>
#include "util/LogMacros.hpp"

int main(int argc, char** argv) {

    Umgebung::util::Logger::instance().init("Umgebung",Umgebung::util::Logger::Level::Trace);

    UMGEBUNG_LOG_INFO("Application started (pid={})", getpid());
    UMGEBUNG_LOG_DEBUG("Debug details: x = {}, y = {}", 42, 3.14);
    UMGEBUNG_LOG_WARN("Warning details: {}", "some warning");
    UMGEBUNG_LOG_ERROR("Error details: {}", "some error");
    UMGEBUNG_LOG_CRIT("Critical details: {}", "some critical thing");
    UMGEBUNG_LOG_TRACE("Tracing details: {}", "doing some tracing");

    return 0;
}