cmake_minimum_required (VERSION 3.21)

set(VCPKG_TARGET_TRIPLET x64-windows)
set(VCPKG_CRT_LINKAGE dynamic)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/submodules/vcpkg/scripts/buildsystems/vcpkg.cmake")

project ("Umgebung" LANGUAGES CXX CUDA VERSION 0.0.1)

option(ENABLE_CODE_ANALYSIS "Use Static Code Analysis on build" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include("${CMAKE_SOURCE_DIR}/cmake/CompilerAndLinker.cmake")

add_executable (${PROJECT_NAME} WIN32 src/walbourn/DebugDraw.cpp
                                      src/walbourn/MSAAHelper.cpp
                                      src/walbourn/RenderTexture.cpp
                                      src/walbourn/SkyboxEffect.cpp
                                      src/walbourn/TextConsole.cpp
							          src/walbourn/DeviceResources.cpp
                                      src/umgebung/Simulation.cpp
							          src/Main.cpp
                                      assets/icon/resource.rc
)

if(NOT MINGW)
    target_precompile_headers(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/include/walbourn/pch.h")
endif()

find_package(directx-dxc CONFIG REQUIRED)
find_package(directx-headers CONFIG REQUIRED)
find_package(directx12-agility CONFIG REQUIRED)
find_package(directxmath CONFIG REQUIRED)
find_package(directxmesh CONFIG REQUIRED)
find_package(directxtex CONFIG REQUIRED)
find_package(directxtk12 CONFIG REQUIRED)
find_package(gameinput CONFIG REQUIRED)
find_package(EnTT CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(unofficial-omniverse-physx-sdk CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(winpixevent CONFIG REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE d3d12.lib
											  dxgi.lib
											  dxguid.lib
											  uuid.lib
											  kernel32.lib
											  user32.lib
											  comdlg32.lib
											  advapi32.lib
											  shell32.lib
											  ole32.lib
											  oleaut32.lib
											  runtimeobject.lib
											  Microsoft::DirectXShaderCompiler
                                              Microsoft::DirectX-Headers
                                              Microsoft::DirectX-Guids
                                              Microsoft::DirectX12-Agility
                                              Microsoft::DirectXMath
                                              Microsoft::DirectXMesh
                                              Microsoft::DirectXTex
                                              Microsoft::DirectXTK12
											  Microsoft::WinPixEventRuntime
                                              EnTT::EnTT
                                              imgui::imgui
                                              nlohmann_json::nlohmann_json
                                              unofficial::omniverse-physx-sdk::sdk
                                              spdlog::spdlog
)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE
         "/wd4061" "/wd4365" "/wd4514" "/wd4571" "/wd4668" "/wd4710" "/wd4820" "/wd5039" "/wd5045"
         "/wd4265" "/wd4625" "/wd4626" "/wd4986" "/wd5204" "/wd5220")

    if(ENABLE_CODE_ANALYSIS)
        target_compile_options(${PROJECT_NAME} PRIVATE /analyze)
    endif()
endif()

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /Wall /EHsc /GR)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE ${COMPILER_DEFINES})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILER_SWITCHES})
target_link_options(${PROJECT_NAME} PRIVATE ${LINKER_SWITCHES})

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>>,copy,true>
    $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}> $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMAND_EXPAND_LISTS
    )

if(TARGET Microsoft::DirectX12-Agility)
   file(MAKE_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/D3D12")
   add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_PROPERTY:Microsoft::DirectX12-Core,IMPORTED_LOCATION_RELEASE> $<TARGET_FILE_DIR:${PROJECT_NAME}>/D3D12
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_PROPERTY:Microsoft::DirectX12-Layers,IMPORTED_LOCATION_DEBUG> $<TARGET_FILE_DIR:${PROJECT_NAME}>/D3D12
      COMMAND ${CMAKE_COMMAND} -E rm -f $<TARGET_FILE_DIR:${PROJECT_NAME}>/D3D12Core.dll
      COMMAND ${CMAKE_COMMAND} -E rm -f $<TARGET_FILE_DIR:${PROJECT_NAME}>/d3d12SDKLayers.dll
      COMMAND_EXPAND_LISTS
      )
endif()

# Optional: import the defined target to copy over the GPU acceleration libraries
# (3rd party provided by NVIDIA)
if(TARGET unofficial::omniverse-physx-sdk::gpu-library)
    if(UNIX)
        # Add rpath setting to find so libraries on unix based systems
        set_target_properties(${PROJECT_NAME} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN"
        )
    endif()
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
else()
    message(WARNING "\GPU acceleration library target not defined
 - GPU acceleration will NOT be available!\
")
endif()