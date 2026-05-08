add_executable(aitrain_core_tests
    ../src/app/src/WorkerClient.h
    ../src/app/src/WorkerClient.cpp
    tst_core.cpp
)

target_include_directories(aitrain_core_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/app/src
)

target_link_libraries(aitrain_core_tests
    PRIVATE
        aitrain_core
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Network
        Qt${QT_VERSION_MAJOR}::Test
        Qt${QT_VERSION_MAJOR}::Sql
)

set(AITRAIN_CORE_TEST_OUTPUT "${CMAKE_BINARY_DIR}/tests/aitrain_core_tests_ctest.txt")
add_test(NAME aitrain_core_tests
    COMMAND aitrain_core_tests -o "${AITRAIN_CORE_TEST_OUTPUT},txt"
)
set_tests_properties(aitrain_core_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${AITRAIN_CORE_TEST_OUTPUT}"
)

aitrain_copy_onnxruntime_runtime(aitrain_core_tests)
aitrain_copy_tensorrt_runtime(aitrain_core_tests)
