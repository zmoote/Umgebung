#pragma once
#include <spdlog/spdlog.h>

namespace umgebung::utils {
    class Logger {
    public:
        static void init();
        static std::shared_ptr<spdlog::logger>& get_logger();
    private:
        static std::shared_ptr<spdlog::logger> logger_;
    };
}