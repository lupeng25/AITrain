add_executable(aitrain_worker
    src/main.cpp
    src/WorkerSession.h
    src/WorkerSession.cpp
    src/WorkerSessionDatasetSplit.cpp
    src/WorkerSessionDataQuality.cpp
    src/WorkerSessionDiagnostics.cpp
    src/WorkerSessionExternalEvidence.cpp
    src/WorkerSessionDatasetConversion.cpp
    src/WorkerSessionDatasetSnapshotImport.cpp
    src/WorkerSessionAnnotation.cpp
    src/WorkerSessionEnvironment.cpp
    src/WorkerSessionModelCommands.cpp
    src/WorkerSessionOcrAcceptance.cpp
    src/WorkerSessionRuntimeDelivery.cpp
    src/WorkerSessionSupport.h
    src/WorkerSessionSupport.cpp
    src/WorkerSessionTrainingWorkflow.cpp
)

target_link_libraries(aitrain_worker
    PRIVATE
        aitrain_foundation
        aitrain_onnx_runtime
        aitrain_application
        aitrain_runtime_matrix
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
