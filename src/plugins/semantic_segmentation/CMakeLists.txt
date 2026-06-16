add_library(SemanticSegmentationPlugin MODULE
    src/SemanticSegmentationPlugin.cpp
    semantic_segmentation.json
)

target_link_libraries(SemanticSegmentationPlugin
    PRIVATE
        aitrain_core
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
)

set_target_properties(SemanticSegmentationPlugin PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins/models
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins/models
)

install(TARGETS SemanticSegmentationPlugin
    LIBRARY DESTINATION plugins/models
    RUNTIME DESTINATION plugins/models
)
