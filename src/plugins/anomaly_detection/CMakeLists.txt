add_library(AnomalyDetectionPlugin MODULE
    src/AnomalyDetectionPlugin.cpp
    anomaly_detection.json
)

target_link_libraries(AnomalyDetectionPlugin
    PRIVATE
        aitrain_core
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
)

set_target_properties(AnomalyDetectionPlugin PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins/models
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins/models
)

install(TARGETS AnomalyDetectionPlugin
    LIBRARY DESTINATION plugins/models
    RUNTIME DESTINATION plugins/models
)
