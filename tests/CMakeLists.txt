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

add_test(NAME aitrain_core_tests COMMAND aitrain_core_tests -silent)

aitrain_copy_onnxruntime_runtime(aitrain_core_tests)
aitrain_copy_tensorrt_runtime(aitrain_core_tests)
