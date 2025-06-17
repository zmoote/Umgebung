cmake_minimum_required(VERSION 3.15)

set(VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/triplets")
set(VCPKG_TARGET_TRIPLET "x64-windows-custom")

# Set project name and language standards
project(Umgebung VERSION 0.1.0 LANGUAGES CXX CUDA)

include(ProcessorCount)
ProcessorCount(NUM_CORES)

if(NOT NUM_CORES EQUAL 0)
  set(CMAKE_BUILD_PARALLEL_LEVEL ${NUM_CORES})
  set(VCPKG_MAX_CONCURRENCY ${NUM_CORES})
  message("CMAKE_BUILD_PARALLEL_LEVEL: ${CMAKE_BUILD_PARALLEL_LEVEL}")
  message("VCPKG_MAX_CONCURRENCY: ${VCPKG_MAX_CONCURRENCY}")
  message("Using ${NUM_CORES} cores for parallel build.")
else()
  set(VCPKG_MAX_CONCURRENCY 0)
  message("Could not determine the number of cores. Building sequentially.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 75 89)

# Enforce debug build type by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

# Platform-specific vcpkg triplet
#if(WIN32)
#    set(VCPKG_TARGET_TRIPLET x64-windows)
#elseif(UNIX)
#    set(VCPKG_TARGET_TRIPLET x64-linux)
#endif()

# Optional: Dynamic CRT linkage
set(VCPKG_CRT_LINKAGE dynamic)

# Executable target
if(WIN32)
    add_executable(${PROJECT_NAME} resources/Umgebung.rc src/main.cpp "src/Core/Application.cpp" "src/Core/Logger.cpp" "src/Core/LayerStack.cpp" "src/Platform/Window.cpp" "src/Renderer/Renderer.cpp" "src/GUI/GuiLayer.cpp")
else()
    add_executable(${PROJECT_NAME} src/main.cpp)
endif()

# Dependencies (vcpkg-managed)
find_package(Stb REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(EnTT CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(vk-bootstrap CONFIG REQUIRED)

# Optional PhysX GPU libs
find_package(unofficial-omniverse-physx-sdk CONFIG REQUIRED)

# Linking dependencies
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        assimp::assimp
        EnTT::EnTT
        glfw
        glm::glm
        imgui::imgui
        spdlog::spdlog
        TBB::tbb
        TBB::tbbmalloc
        TBB::tbbmalloc_proxy
        vk-bootstrap::vk-bootstrap
        vk-bootstrap::vk-bootstrap-compiler-warnings
        unofficial::omniverse-physx-sdk::sdk
)

# Optional: Copy GPU libs if available
if(TARGET unofficial::omniverse-physx-sdk::gpu-library)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:unofficial::omniverse-physx-sdk::gpu-library>
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)

    if(WIN32)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:unofficial::omniverse-physx-sdk::gpu-device-library>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    endif()

    if(UNIX)
        set_target_properties(${PROJECT_NAME} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN"
        )
    endif()
else()
    message(WARNING "GPU acceleration library target not defined — GPU acceleration will NOT be available!")
endif()

# Copy resources directory to build output
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/resources
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/resources
)

# Shader compilation
find_program(GLSL_VALIDATOR glslangValidator
    HINTS
        ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}
        /usr/bin /usr/local/bin
        $ENV{VULKAN_SDK}/Bin $ENV{VULKAN_SDK}/Bin32
)

if(NOT GLSL_VALIDATOR)
    message(FATAL_ERROR "glslangValidator not found! Install Vulkan SDK or ensure shader tools are available.")
endif()

# Collect shader sources
file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.frag"
    "${PROJECT_SOURCE_DIR}/shaders/*.vert"
)

# Compile shaders to SPIR-V in binary build dir
set(SHADER_BINARY_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    set(SPIRV "${SHADER_BINARY_DIR}/${FILE_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
        DEPENDS ${GLSL}
    )
    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach()

# Shader build target
add_custom_target(Shaders DEPENDS ${SPIRV_BINARY_FILES})

# Set validation layers flag based on build configuration
target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:ENABLE_VULKAN_VALIDATION_LAYERS=1>
    $<$<NOT:$<CONFIG:Debug>>:ENABLE_VULKAN_VALIDATION_LAYERS=0>
)

# Automatically set VK_ADD_LAYER_PATH when debugging in Visual Studio
if (MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(VK_LAYER_BIN_PATH "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows-custom/bin")
    set(VK_LAYER_JSON_SRC "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows-custom/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json")
    set(VK_LAYER_JSON_DEST "${CMAKE_BINARY_DIR}/vulkan/explicit_layer.d")
    
    file(MAKE_DIRECTORY ${VK_LAYER_JSON_DEST})
    
    # Check if JSON file exists before copying
    if(EXISTS "${VK_LAYER_JSON_SRC}")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${VK_LAYER_JSON_SRC}"
            "${VK_LAYER_JSON_DEST}/VkLayer_khronos_validation.json"
            COMMENT "Copying Vulkan validation layer JSON to ${VK_LAYER_JSON_DEST}"
        )
    else()
        message(WARNING "Vulkan validation layer JSON not found at ${VK_LAYER_JSON_SRC}. Ensure vulkan-validationlayers is installed via vcpkg.")
    endif()
    
    set_target_properties(${PROJECT_NAME} PROPERTIES
        VS_DEBUGGER_ENVIRONMENT "VK_LAYER_PATH=${VK_LAYER_JSON_DEST}"
    )

    message(STATUS "VK_LAYER_PATH set for Visual Studio debugger to: ${VK_LAYER_JSON_DEST}")
endif()

add_dependencies(${PROJECT_NAME} Shaders)