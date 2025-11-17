cmake_minimum_required (VERSION 3.21)

set(VCPKG_TARGET_TRIPLET x64-windows)
set(VCPKG_CRT_LINKAGE dynamic)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/submodules/vcpkg/scripts/buildsystems/vcpkg.cmake")

project ("Umgebung" LANGUAGES CXX CUDA VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES 75 89)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable (${PROJECT_NAME} assets/icon/Umgebung.rc
                                src/Main.cpp
                                src/app/Application.cpp
                                src/ecs/entities/Galaxy.cpp
                                src/ecs/entities/Multiverse.cpp
                                src/ecs/entities/Planet.cpp
                                src/ecs/entities/SolarSystem.cpp
                                src/ecs/entities/Star.cpp
                                src/ecs/entities/Universe.cpp
                                src/renderer/Renderer.cpp
                                src/ui/Window.cpp
                                src/util/Logger.cpp
                                src/ui/imgui/Panel.cpp
                                src/ui/imgui/StatisticsPanel.cpp
                                src/ui/imgui/HierarchyPanel.cpp
                                src/ui/imgui/AboutPanel.cpp
                                src/ecs/components/Consciousness.cpp
                                src/ecs/components/Soul.cpp
                                src/renderer/Camera.cpp
                                src/renderer/gl/Shader.cpp
                                src/renderer/Renderer.cpp
                                src/ui/imgui/ViewportPanel.cpp
                                src/renderer/Framebuffer.cpp
                                src/ui/imgui/PropertiesPanel.cpp
                                src/ui/imgui/FilePickerPanel.cpp
                                src/ui/imgui/ConsolePanel.cpp
                                src/ecs/components/Transform.cpp
                                src/renderer/Mesh.cpp
                                src/scene/Scene.cpp
                                src/ecs/systems/RenderSystem.cpp
                                src/ecs/systems/AssetSystem.cpp
                                src/ecs/systems/PhysicsSystem.cpp
                                src/ui/UIManager.cpp
                                src/scene/SceneSerializer.cpp
                                src/asset/ModelLoader.cpp
)

find_package(assimp CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(EnTT CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Stb REQUIRED)

# PhysX
set(PHYSX_VCPKG_DIR "${CMAKE_SOURCE_DIR}/submodules/vcpkg/packages/physx_x64-windows")
set(PHYSX_INCLUDE_DIR "${PHYSX_VCPKG_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PHYSX_INCLUDE_DIR}
    ${PHYSX_INCLUDE_DIR}/physx
)

set(PHYSX_LIBS
    PhysX_64
    PhysXCommon_64
    PhysXCooking_64
    PhysXFoundation_64
    PhysXExtensions_static_64
    PhysXTask_static_64
    PhysXPvdSDK_static_64
)

foreach(LIB ${PHYSX_LIBS})
    target_link_libraries(${PROJECT_NAME} PRIVATE
        debug "${PHYSX_VCPKG_DIR}/debug/lib/${LIB}.lib"
        optimized "${PHYSX_VCPKG_DIR}/lib/${LIB}.lib"
    )
endforeach()

target_include_directories(${PROJECT_NAME} PRIVATE ${Stb_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE assimp::assimp
                                              CUDA::cudart
                                              Eigen3::Eigen
                                              EnTT::EnTT
                                              glad::glad
                                              glfw
                                              glm::glm
                                              imgui::imgui
                                              nlohmann_json::nlohmann_json
                                              spdlog::spdlog
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>>,copy,true>
    $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}> $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMAND_EXPAND_LISTS
    )

    add_custom_command(
    TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/assets
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/assets
    COMMENT "Copying assets directory to build output"
)

# Copy PhysX GPU libraries
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PHYSX_VCPKG_DIR}/bin/PhysXGpu_64.dll"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying PhysX GPU release DLL"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PHYSX_VCPKG_DIR}/debug/bin/PhysXGpu_64.dll"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying PhysX GPU debug DLL"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PHYSX_VCPKG_DIR}/bin/PhysXDevice64.dll"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying PhysX Device release DLL"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PHYSX_VCPKG_DIR}/debug/bin/PhysXDevice64.dll"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying PhysX Device debug DLL"
)