#pragma once
#include <memory>
#include <spdlog/spdlog.h>

namespace Umgebung {

    class Logger {
    public:
        static void Init();
        static std::shared_ptr<spdlog::logger>& GetCoreLogger();

    private:
        static std::shared_ptr<spdlog::logger> coreLogger;
    };

} // namespace Umgebung