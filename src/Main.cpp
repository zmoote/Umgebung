/**
 * @file Main.cpp
 * @brief Process entry point for the Umgebung Reality Simulation Engine.
 *
 * @details
 * Delegates all startup CLI parsing, hardware discovery/adaptation, optional
 * desktop GUI windowing (`UMGEBUNG_HAS_GUI == 1`), and main-loop orchestration
 * to `umgebung::core::Application`. Keeping `main()` minimal ensures that the
 * exact same application lifecycle can also be exercised programmatically in
 * unit and integration tests.
 */

#include "umgebung/core/Application.hpp"

#include <cstdlib>

/**
 * @brief Operating system entry point for the `Umgebung` executable.
 *
 * @param argc Total number of command-line arguments supplied by the shell/launcher.
 * @param argv Null-terminated array of CLI argument C-strings.
 *
 * @return `EXIT_SUCCESS` (`0`) on clean termination or informational early exit
 *         (`--help` / `--version`); `EXIT_FAILURE` (`1`) or non-zero solver
 *         error code if initialization or execution fails.
 */
int main(int argc, char** argv) {
    umgebung::core::Application app;

    const umgebung::core::ApplicationInitResult initStatus = app.Initialize(argc, argv);
    if (initStatus == umgebung::core::ApplicationInitResult::EarlyExitSuccess) {
        return EXIT_SUCCESS;
    }
    if (initStatus == umgebung::core::ApplicationInitResult::Failure) {
        return EXIT_FAILURE;
    }

    const int exitCode = app.Run();
    app.Shutdown();
    return exitCode;
}
