add_executable(aitrain_worker
    src/main.cpp
    src/WorkerSession.h
    src/WorkerSession.cpp
    src/WorkerSessionCommandRegistry.cpp
    src/WorkerSessionDatasetSplitV2.cpp
    src/WorkerSessionDataQualityV2.cpp
    src/WorkerSessionDiagnosticsV2.cpp
    src/WorkerSessionDatasetConversionV2.cpp
    src/WorkerSessionDatasetSnapshotImportV2.cpp
    src/WorkerSessionAnnotationV2.cpp
    src/WorkerSessionEnvironment.cpp
    src/WorkerSessionModelCommands.cpp
    src/WorkerSessionOcrAcceptanceV2.cpp
    src/WorkerSessionRuntimeDeliveryV2.cpp
    src/WorkerSessionSupport.h
    src/WorkerSessionSupport.cpp
    src/WorkerSessionTrainingWorkflowV2.cpp
)

target_link_libraries(aitrain_worker
    PRIVATE
        aitrain_core
        aitrain_v2_onnx_runtime
        aitrain_v2_application
        aitrain_v2_runtime_matrix
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Network
)

set_target_properties(aitrain_worker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

aitrain_copy_onnxruntime_runtime(aitrain_worker)
aitrain_copy_ncnn_runtime(aitrain_worker)
aitrain_copy_tensorrt_runtime(aitrain_worker)
aitrain_copy_qt_platform_runtime(aitrain_worker)

install(TARGETS aitrain_worker
    RUNTIME DESTINATION .
)
