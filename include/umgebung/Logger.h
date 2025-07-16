#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <memory>
#include <string>
#include <iostream>

namespace Umgebung {

    class Logger {
    public:
        // Delete copy constructor and assignment operator for singleton
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        // Get singleton instance
        static Logger& getInstance() {
            static Logger instance;
            return instance;
        }

        // Log level enum for easier configuration
        enum class Level {
            Trace,
            Debug,
            Info,
            Warn,
            Error,
            Critical,
            Off
        };

        // Initialize logger with output file and log level
        void initialize(const std::string& logFile = "umgebung.log",
            Level level = Level::Info) {
            try {
                // Create console sink
                auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
                console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v");

                // Create file sink with 5MB size limit and 3 rotated files
                auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                    logFile, 1024 * 1024 * 5, 3);
                file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v");

                // Create logger with multiple sinks
                std::vector<spdlog::sink_ptr> sinks{ console_sink, file_sink };
                logger_ = std::make_shared<spdlog::logger>("Umgebung", sinks.begin(), sinks.end());

                // Register logger
                spdlog::register_logger(logger_);

                // Set log level
                setLevel(level);

                // Enable backtrace for critical errors
                logger_->enable_backtrace(32);

                // Set flush policy
                logger_->flush_on(spdlog::level::err);
                spdlog::flush_every(std::chrono::seconds(3));

                logger_->info("Logger initialized successfully");
            }
            catch (const spdlog::spdlog_ex& ex) {
                std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
            }
        }

        // Set log level
        void setLevel(Level level) {
            switch (level) {
            case Level::Trace:
                logger_->set_level(spdlog::level::trace);
                break;
            case Level::Debug:
                logger_->set_level(spdlog::level::debug);
                break;
            case Level::Info:
                logger_->set_level(spdlog::level::info);
                break;
            case Level::Warn:
                logger_->set_level(spdlog::level::warn);
                break;
            case Level::Error:
                logger_->set_level(spdlog::level::err);
                break;
            case Level::Critical:
                logger_->set_level(spdlog::level::critical);
                break;
            case Level::Off:
                logger_->set_level(spdlog::level::off);
                break;
            }
        }

        // Logging methods
        template<typename... Args>
        void trace(const Args&... args) {
            logger_->trace(args...);
        }

        template<typename... Args>
        void debug(const Args&... args) {
            logger_->debug(args...);
        }

        template<typename... Args>
        void info(const Args&... args) {
            logger_->info(args...);
        }

        template<typename... Args>
        void warn(const Args&... args) {
            logger_->warn(args...);
        }

        template<typename... Args>
        void error(const Args&... args) {
            logger_->error(args...);
        }

        template<typename... Args>
        void critical(const Args&... args) {
            logger_->critical(args...);
            logger_->dump_backtrace();
        }

    private:
        Logger() = default;
        std::shared_ptr<spdlog::logger> logger_;
    };

} // namespace Umgebung

// Convenience macros for logging
#define UMB_LOG_TRACE(...) Umgebung::Logger::getInstance().trace(__VA_ARGS__)
#define UMB_LOG_DEBUG(...) Umgebung::Logger::getInstance().debug(__VA_ARGS__)
#define UMB_LOG_INFO(...) Umgebung::Logger::getInstance().info(__VA_ARGS__)
#define UMB_LOG_WARN(...) Umgebung::Logger::getInstance().warn(__VA_ARGS__)
#define UMB_LOG_ERROR(...) Umgebung::Logger::getInstance().error(__VA_ARGS__)
#define UMB_LOG_CRITICAL(...) Umgebung::Logger::getInstance().critical(__VA_ARGS__)