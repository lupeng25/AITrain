add_library(YoloNativePlugin MODULE
    src/YoloNativePlugin.cpp
    yolo_native.json
)

target_link_libraries(YoloNativePlugin
    PRIVATE
        aitrain_core
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
)

set_target_properties(YoloNativePlugin PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins/models
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins/models
)

install(TARGETS YoloNativePlugin
    LIBRARY DESTINATION plugins/models
    RUNTIME DESTINATION plugins/models
)
