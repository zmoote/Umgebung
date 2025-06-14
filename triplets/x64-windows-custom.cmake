set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)

# Enable SPDLOG_WCHAR_FILENAMES for spdlog
if(PORT MATCHES "spdlog")
    set(SPDLOG_WCHAR_FILENAMES ON)
endif()

# Workaround for Vulkan Validation Layers compiler error
if(PORT MATCHES "vulkan-validationlayers")
    set(VCPKG_CXX_FLAGS "${VCPKG_CXX_FLAGS} /d2ssaopt-beforevect- /Od")
    set(VCPKG_C_FLAGS "${VCPKG_C_FLAGS} /d2ssaopt-beforevect- /Od")
    # Explicitly disable /EHsc to test compatibility
    string(REGEX REPLACE "/EHsc" "" VCPKG_CXX_FLAGS "${VCPKG_CXX_FLAGS}")
    string(REGEX REPLACE "/EHsc" "" VCPKG_C_FLAGS "${VCPKG_C_FLAGS}")
endif()