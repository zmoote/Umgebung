find_package(GTest CONFIG REQUIRED)

add_executable(umgebung_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_engine.cpp
)

target_link_libraries(umgebung_test PRIVATE umgebung_core GTest::gtest GTest::gmock)

add_test(NAME umgebung_test COMMAND umgebung_test)
