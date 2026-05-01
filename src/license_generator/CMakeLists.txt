add_executable(AITrainLicenseGenerator WIN32
    main.cpp
    LicenseGeneratorWindow.h
    LicenseGeneratorWindow.cpp
    ../app/src/AppStyle.h
    ../app/src/AppStyle.cpp
)

target_include_directories(AITrainLicenseGenerator
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/app/src
)

target_link_libraries(AITrainLicenseGenerator
    PRIVATE
        aitrain_core
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Widgets
)

set_target_properties(AITrainLicenseGenerator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

if(AITRAIN_INSTALL_LICENSE_GENERATOR)
    install(TARGETS AITrainLicenseGenerator
        RUNTIME DESTINATION tools
    )
endif()
