// --- Core/Logger.cpp ---
#include "Logger.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>

namespace Umgebung {

    std::shared_ptr<spdlog::logger> Logger::coreLogger;

    void Logger::Init() {
        spdlog::set_pattern("[%T] %^%l%$ | %v");
        coreLogger = spdlog::stdout_color_mt("UMG");
        coreLogger->set_level(spdlog::level::trace);
    }

    std::shared_ptr<spdlog::logger>& Logger::GetCoreLogger() {
        return coreLogger;
    }

} // namespace Umgebung