add_library(DatasetInteropPlugin MODULE
    src/DatasetInteropPlugin.cpp
    dataset_interop.json
)

target_link_libraries(DatasetInteropPlugin
    PRIVATE
        aitrain_core
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
)

set_target_properties(DatasetInteropPlugin PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins/models
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins/models
)

