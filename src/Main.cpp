#include "umgebung/core/EngineContext.hpp"
#include "umgebung/graphics/RenderDevice.hpp"
#include <spdlog/spdlog.h>

#if UMGEBUNG_HAS_GUI
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#endif

int main(int argc, char** argv) {
	spdlog::info("Starting Umgebung Simulation Engine v0.0.1");

	// 1. Initialize Engine Context (which initializes RenderDevice if GUI is active)
	umgebung::core::EngineContext engineContext;
	if (!engineContext.Initialize(argc, argv)) {
		spdlog::error("Failed to initialize EngineContext.");
		return -1;
	}

	const auto& caps = engineContext.GetHardwareCapabilities();
	spdlog::info("Hardware profile loaded: GPU = {}, Compute = {}.{}",
		caps.gpuName, caps.computeCapabilityMajor, caps.computeCapabilityMinor);

#if UMGEBUNG_HAS_GUI
	// Retrieve the RenderDevice managed by EngineContext instead of creating a new one
	const auto* renderDevice = engineContext.GetRenderDevice();
	if (!renderDevice) {
		spdlog::error("RenderDevice is null.");
		return -1;
	}
	spdlog::info("GUI / RenderDevice initialization successful. Entering main loop...");

	// 2. Main Window / Simulation Loop
	GLFWwindow* window = renderDevice->GetWindow();
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		// TODO: Call renderDevice->BeginFrame(), draw simulation / ImGui, renderDevice->Present()
	}
#else
	spdlog::info("Running in headless HPC mode.");
#endif

	// 3. Clean shutdown via EngineContext
	engineContext.Shutdown();
	spdlog::info("Initialization phase successful. Shutting down.");
	return 0;
}