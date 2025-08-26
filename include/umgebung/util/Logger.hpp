#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>
#include <string>
#include <mutex>

namespace Umgebung::util {

    /**
     * @brief Centralised, thread‑safe logger.
     *
     * The class is a singleton – initialise once via `init()` and
     * then use the convenience wrappers.
     */
    class Logger {
    public:
        enum class Level { Trace, Debug, Info, Warn, Error, Critical, Off };

        // Delete copy/move – only one instance exists
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        /// Global access point
        static Logger& instance();

        /// Initialise the logger – idempotent and thread‑safe
        void init(const std::string& name = "umgebung",
            Level level = Level::Info,
            bool enableConsole = true,
            bool enableFile = true,
            const std::string& filePath = "umgebung.log");

        /* ------------------------------------------------------------
           Convenience wrappers that forward directly to spdlog
        ------------------------------------------------------------ */
        template<class... Args>
        void trace(const std::string& fmt, const Args&... args) {
            m_logger->trace(fmt, args...);
        }
        template<class... Args>
        void debug(const std::string& fmt, const Args&... args) {
            m_logger->debug(fmt, args...);
        }
        template<class... Args>
        void info(const std::string& fmt, const Args&... args) {
            m_logger->info(fmt, args...);
        }
        template<class... Args>
        void warn(const std::string& fmt, const Args&... args) {
            m_logger->warn(fmt, args...);
        }
        template<class... Args>
        void error(const std::string& fmt, const Args&... args) {
            m_logger->error(fmt, args...);
        }
        template<class... Args>
        void critical(const std::string& fmt, const Args&... args) {
            m_logger->critical(fmt, args...);
        }

        /** Return the underlying spdlog logger – use sparingly. */
        std::shared_ptr<spdlog::logger> underlying() const { return m_logger; }

    private:
        Logger() = default;
        std::shared_ptr<spdlog::logger> m_logger;
        std::mutex m_initMutex;
    };

} // namespace Umgebung::util