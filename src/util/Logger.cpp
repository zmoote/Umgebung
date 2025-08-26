#include "umgebung/util/Logger.hpp"

namespace Umgebung::util {

    Logger& Logger::instance() {
        static Logger inst;          // thread‑safe since C++11
        return inst;
    }

    void Logger::init(const std::string& name,
        Level level,
        bool enableConsole,
        bool enableFile,
        const std::string& filePath)
    {
        std::lock_guard<std::mutex> lock(m_initMutex);

        if (m_logger) return;   // already initialised

        std::vector<spdlog::sink_ptr> sinks;

        if (enableConsole) {
            auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console->set_pattern("%^[%T] %n: %v%$");
            sinks.push_back(console);
        }

        if (enableFile) {
            auto file = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filePath, true);
            file->set_pattern("[%T] [%l] %v");
            sinks.push_back(file);
        }

        m_logger = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());

        // Map our enum → spdlog enum
        m_logger->set_level(static_cast<spdlog::level::level_enum>(level));

        // Make the logger globally accessible if someone else calls
        // spdlog::get(name).  (Optional – keep it for compatibility.)
        spdlog::register_logger(m_logger);
    }

} // namespace Umgebung::util