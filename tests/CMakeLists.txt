add_executable(aitrain_core_tests
    tst_core.cpp
)

target_link_libraries(aitrain_core_tests
    PRIVATE
        aitrain_core
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Test
        Qt${QT_VERSION_MAJOR}::Sql
)

add_test(NAME aitrain_core_tests COMMAND aitrain_core_tests)

