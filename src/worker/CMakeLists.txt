add_executable(aitrain_worker
    src/main.cpp
    src/WorkerSession.h
    src/WorkerSession.cpp
    src/WorkerSessionDatasetCommands.cpp
    src/WorkerSessionEnvironment.cpp
    src/WorkerSessionModelCommands.cpp
    src/WorkerSessionPipeline.cpp
    src/WorkerSessionPythonTrainer.cpp
    src/WorkerSessionSupport.h
    src/WorkerSessionSupport.cpp
    src/WorkerSessionTraining.cpp
)

target_link_libraries(aitrain_worker
    PRIVATE
        aitrain_core
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Network
)

set_target_properties(aitrain_worker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

aitrain_copy_onnxruntime_runtime(aitrain_worker)
aitrain_copy_tensorrt_runtime(aitrain_worker)

install(TARGETS aitrain_worker
    RUNTIME DESTINATION .
)
