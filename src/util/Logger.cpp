#include "Logger.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

namespace Umgebung {
    namespace util {

        std::shared_ptr<spdlog::logger> Logger::s_logger = nullptr;

        void Logger::Init(const std::string& configPath)
        {
            if (s_logger) {
                // Already initialised – ignore or log a warning
                s_logger->warn("Logger::Init called more than once.");
                return;
            }

            /* ---------- Load configuration ---------- */
            std::string levelStr = "info";
            std::string fileName = "umgebung.log";
            std::size_t  maxSize = 10 * 1024 * 1024; // 10 MiB
            std::size_t  maxFiles = 5;

            if (!configPath.empty()) {
                std::ifstream in(configPath);
                if (in) {
                    try {
                        nlohmann::json cfg;
                        in >> cfg;

                        if (cfg.contains("level"))
                            levelStr = cfg.at("level").get<std::string>();

                        if (cfg.contains("filename"))
                            fileName = cfg.at("filename").get<std::string>();

                        if (cfg.contains("max_size"))
                            maxSize = cfg.at("max_size").get<std::size_t>();

                        if (cfg.contains("max_files"))
                            maxFiles = cfg.at("max_files").get<std::size_t>();
                    }
                    catch (const std::exception& e) {
                        std::cerr << "[Logger] Config parse error: " << e.what() << '\n';
                    }
                }
                else {
                    std::cerr << "[Logger] Could not open config file: " << configPath << '\n';
                }
            }

            /* ---------- Create sinks ---------- */
            std::vector<spdlog::sink_ptr> sinks;
            sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
            sinks.emplace_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(fileName, maxSize, maxFiles));

            /* ---------- Construct logger ---------- */
            s_logger = std::make_shared<spdlog::logger>("umgebung", begin(sinks), end(sinks));

            // Set pattern: timestamp | level | message
            s_logger->set_pattern("%Y-%m-%d %H:%M:%S.%e | %-8l | %v");

            // Map string to level
            spdlog::level::level_enum lvl = spdlog::level::info;
            if (levelStr == "trace")   lvl = spdlog::level::trace;
            if (levelStr == "debug")   lvl = spdlog::level::debug;
            if (levelStr == "info")    lvl = spdlog::level::info;
            if (levelStr == "warn")    lvl = spdlog::level::warn;
            if (levelStr == "error")   lvl = spdlog::level::err;
            if (levelStr == "critical")lvl = spdlog::level::critical;

            s_logger->set_level(lvl);
            s_logger->flush_on(lvl);   // flush immediately on critical or higher

            // Register with spdlog to enable automatic shutdown
            spdlog::register_logger(s_logger);
        }

        std::shared_ptr<spdlog::logger> Logger::Get()
        {
            if (!s_logger) {
                // Fallback: create a minimal logger to avoid crashes
                Init(); // defaults
            }
            return s_logger;
        }

    } // namespace util
} // namespace Umgebung