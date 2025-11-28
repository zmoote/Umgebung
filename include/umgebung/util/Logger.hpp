/**
 * @file Logger.hpp
 * @brief Contains the Logger class.
 */
#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include "umgebung/ui/imgui/ImGuiConsoleSink.hpp"
#include <memory>
#include <string>
#include <mutex>

namespace Umgebung::util {
    /**
     * @brief A singleton logger class that uses spdlog.
     */
    class Logger {
    public:
        /**
         * @brief The log level.
         */
        enum class Level { Trace, Debug, Info, Warn, Error, Critical, Off };

        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        /**
         * @brief Get the static instance of the logger.
         * 
         * @return Logger& 
         */
        static Logger& instance();

        /**
         * @brief Initializes the logger.
         * 
         * @param name The name of the logger.
         * @param level The log level.
         * @param enableConsole Whether to enable console logging.
         * @param enableFile Whether to enable file logging.
         * @param enableConsolePanel Whether to enable the ImGui console panel sink.
         * @param filePath The path to the log file.
         */
        void init(const std::string& name = "umgebung",
            Level level = Level::Info,
            bool enableConsole = true,
            bool enableFile = true,
            bool enableConsolePanel = true,
            const std::string& filePath = "logs/umgebung.log");

        /**
         * @brief Logs a trace message.
         * 
         * @tparam Args 
         * @param fmt The format string.
         * @param args The arguments.
         */
        template<class... Args>
        void trace(const std::string& fmt, const Args&... args) {
            m_logger->trace(fmt, args...);
        }

        /**
         * @brief Logs a debug message.
         * 
         * @tparam Args 
         * @param fmt The format string.
         * @param args The arguments.
         */
        template<class... Args>
        void debug(const std::string& fmt, const Args&... args) {
            m_logger->debug(fmt, args...);
        }

        /**
         * @brief Logs an info message.
         * 
         * @tparam Args 
         * @param fmt The format string.
         * @param args The arguments.
         */
        template<class... Args>
        void info(const std::string& fmt, const Args&... args) {
            m_logger->info(fmt, args...);
        }

        /**
         * @brief Logs a warning message.
         * 
         * @tparam Args 
         * @param fmt The format string.
         * @param args The arguments.
         */
        template<class... Args>
        void warn(const std::string& fmt, const Args&... args) {
            m_logger->warn(fmt, args...);
        }

        /**
         * @brief Logs an error message.
         * 
         * @tparam Args 
         * @param fmt The format string.
         * @param args The arguments.
         */
        template<class... Args>
        void error(const std::string& fmt, const Args&... args) {
            m_logger->error(fmt, args...);
        }

        /**
         * @brief Logs a critical message.
         * 
         * @tparam Args 
         * @param fmt The format string.
         * @param args The arguments.
         */
        template<class... Args>
        void critical(const std::string& fmt, const Args&... args) {
            m_logger->critical(fmt, args...);
        }

        /**
         * @brief Get the underlying spdlog logger.
         * 
         * @return std::shared_ptr<spdlog::logger> 
         */
        std::shared_ptr<spdlog::logger> underlying() const { return m_logger; }

        /**
         * @brief Get the Panel Sink Buffer object.
         * 
         * @return const std::vector<std::string>& 
         */
        const std::vector<std::string>& getPanelSinkBuffer() const;

        /**
         * @brief Clears the panel sink buffer.
         */
        void clearPanelSinkBuffer();

    private:
        Logger() = default;
        std::shared_ptr<spdlog::logger> m_logger; ///< The spdlog logger.
        std::mutex m_initMutex;                   ///< The mutex for initializing the logger.

        std::shared_ptr<ui::imgui::ImGuiConsoleSink_mt> m_panelSink; ///< The ImGui console panel sink.
    };

}