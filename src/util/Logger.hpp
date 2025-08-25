#pragma once

#include <string>
#include <memory>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

namespace Umgebung {
    namespace util {

        /**
         * @brief Centralised logger.
         *
         * Usage:
         *   Umgebung::util::Logger::Init("config.json");   // once, before any log calls
         *   Umgebung::util::Logger::Get()->info("Hello, {}!", "world");
         */
        class Logger
        {
        public:
            // Delete copy / move to enforce singleton-like access
            Logger(const Logger&) = delete;
            Logger& operator=(const Logger&) = delete;
            Logger(Logger&&) noexcept = delete;
            Logger& operator=(Logger&&) noexcept = delete;

            /**
             * @brief Initialise the logger from a JSON config.
             * @param configPath Path to a JSON file with optional keys:
             *                   - "level" (string, default "info")
             *                   - "filename" (string, default "umgebung.log")
             *                   - "max_size" (size_t, bytes, default 10MiB)
             *                   - "max_files" (size_t, default 5)
             *
             * Must be called once before any other `Get()` call.
             */
            static void Init(const std::string& configPath = "");

            /**
             * @brief Return a shared pointer to the underlying spdlog logger.
             */
            static std::shared_ptr<spdlog::logger> Get();

        private:
            // Hidden constructor – only Init() creates the instance.
            Logger() = default;

            static std::shared_ptr<spdlog::logger> s_logger;
        };

    } // namespace util
} // namespace Umgebung