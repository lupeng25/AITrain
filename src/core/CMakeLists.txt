add_library(aitrain_core STATIC
    include/aitrain/core/PluginInterfaces.h
    include/aitrain/core/DatasetValidators.h
    include/aitrain/core/DetectionDataset.h
    include/aitrain/core/DetectionTrainer.h
    include/aitrain/core/OcrRecDataset.h
    include/aitrain/core/OcrRecTrainer.h
    include/aitrain/core/SegmentationDataset.h
    include/aitrain/core/SegmentationTrainer.h
    include/aitrain/core/JsonProtocol.h
    include/aitrain/core/PluginManager.h
    include/aitrain/core/ProjectRepository.h
    include/aitrain/core/TaskModels.h
    src/JsonProtocol.cpp
    src/DatasetValidators.cpp
    src/DetectionDataset.cpp
    src/DetectionTrainer.cpp
    src/OcrRecDataset.cpp
    src/OcrRecTrainer.cpp
    src/SegmentationDataset.cpp
    src/SegmentationTrainer.cpp
    src/PluginManager.cpp
    src/ProjectRepository.cpp
    src/TaskModels.cpp
)

target_include_directories(aitrain_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_core
    PUBLIC
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Network
        Qt${QT_VERSION_MAJOR}::Sql
)

target_compile_definitions(aitrain_core
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

if(AITRAIN_WITH_ONNXRUNTIME)
    target_compile_definitions(aitrain_core
        PUBLIC
            AITRAIN_WITH_ONNXRUNTIME
    )
    target_include_directories(aitrain_core
        PRIVATE
            ${AITRAIN_ONNXRUNTIME_INCLUDE_DIR}
    )
    target_link_libraries(aitrain_core
        PRIVATE
            ${AITRAIN_ONNXRUNTIME_LIBRARY}
    )
endif()
