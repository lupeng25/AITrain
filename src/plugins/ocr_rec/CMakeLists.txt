add_library(OcrRecNativePlugin MODULE
    src/OcrRecNativePlugin.cpp
    ocr_rec.json
)

target_link_libraries(OcrRecNativePlugin
    PRIVATE
        aitrain_core
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
)

set_target_properties(OcrRecNativePlugin PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins/models
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins/models
)

