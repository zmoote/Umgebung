# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

set(VCPKG_TARGET_TRIPLET x64-windows)
set(VCPKG_CRT_LINKAGE dynamic)

project(Umgebung)

add_executable(${PROJECT_NAME} src/main.cpp)

# Note: if the package cannot be found here, check that you're using the right triplet
find_package(unofficial-omniverse-physx-sdk CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE unofficial::omniverse-physx-sdk::sdk)

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
