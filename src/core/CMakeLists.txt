add_library(aitrain_v2_domain STATIC
    include/aitrain/v2/DomainTypes.h
    src/v2/DomainTypes.cpp
)

target_include_directories(aitrain_v2_domain
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_v2_domain
    PUBLIC
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_v2_domain
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_v2_protocol STATIC
    include/aitrain/v2/ProtocolV2.h
    src/v2/ProtocolV2.cpp
)

target_include_directories(aitrain_v2_protocol
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_v2_protocol
    PUBLIC
        aitrain_v2_domain
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_v2_protocol
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_v2_storage STATIC
    include/aitrain/v2/StorageV2.h
    src/v2/StorageV2.cpp
)

add_library(aitrain_v2_training_workflow_profile STATIC
    include/aitrain/v2/TrainingWorkflowProfileV2.h
    src/v2/TrainingWorkflowProfileV2.cpp
)

target_include_directories(aitrain_v2_training_workflow_profile
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_v2_training_workflow_profile
    PUBLIC
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_v2_training_workflow_profile
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

target_include_directories(aitrain_v2_storage
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_v2_storage
    PUBLIC
        aitrain_v2_domain
        aitrain_v2_model
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql
)

target_compile_definitions(aitrain_v2_storage
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_v2_application STATIC
    include/aitrain/v2/CapabilityPlannerV2.h
    include/aitrain/v2/TaskCoordinator.h
    include/aitrain/v2/TaskExecutionHostV2.h
    include/aitrain/v2/WorkflowRunnerV2.h
    include/aitrain/v2/EvidenceBundleV2.h
    include/aitrain/v2/EvidenceRendererV2.h
    include/aitrain/v2/ModelImportServiceV2.h
    include/aitrain/v2/ModelPackageRuntimeServiceV2.h
    include/aitrain/v2/ProjectWorkspaceV2.h
    include/aitrain/v2/ProjectQueryServiceV2.h
    src/v2/CapabilityPlannerV2.cpp
    src/v2/TaskCoordinator.cpp
    src/v2/TaskExecutionHostV2.cpp
    src/v2/WorkflowRunnerV2.cpp
    src/v2/EvidenceBundleV2.cpp
    src/v2/EvidenceRendererV2.cpp
    src/v2/ModelImportServiceV2.cpp
    src/v2/ModelPackageRuntimeServiceV2.cpp
    src/v2/ProjectWorkspaceV2.cpp
    src/v2/ProjectWorkspaceAnnotationV2.cpp
    src/v2/ProjectWorkspaceDataQualityV2.cpp
    src/v2/ProjectWorkspaceDiagnosticsV2.cpp
    src/v2/ProjectWorkspaceDatasetConversionV2.cpp
    src/v2/ProjectWorkspaceDatasetSnapshotImportV2.cpp
    src/v2/ProjectWorkspaceDatasetSplitV2.cpp
    src/v2/ProjectWorkspaceOcrAcceptanceV2.cpp
    src/v2/ProjectWorkspaceOcrReportPackagerV2.cpp
    src/v2/ProjectWorkspaceRuntimeDeliveryV2.cpp
    src/v2/ProjectQueryServiceV2.cpp
)

add_library(aitrain_v2_dataset_conversion STATIC
    include/aitrain/v2/DatasetConversionServiceV2.h
    src/v2/DatasetConversionServiceV2.cpp
)

target_include_directories(aitrain_v2_dataset_conversion PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_v2_dataset_conversion
    PUBLIC
        aitrain_v2_artifact
        aitrain_v2_dataset
        aitrain_core
        Qt${QT_VERSION_MAJOR}::Core
)
target_compile_definitions(aitrain_v2_dataset_conversion PUBLIC QT_NO_CAST_FROM_ASCII)

target_include_directories(aitrain_v2_application
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_v2_application
    PUBLIC
        aitrain_v2_training_workflow_profile
        aitrain_v2_protocol
        aitrain_v2_process
        aitrain_v2_storage
        aitrain_v2_artifact
        aitrain_v2_runtime_matrix
        aitrain_v2_onnx_runtime
        aitrain_v2_native_runtimes
        aitrain_v2_dataset
        aitrain_v2_dataset_conversion
        aitrain_core
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_v2_application
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_v2_artifact STATIC
    include/aitrain/v2/ArtifactStoreV2.h
    src/v2/ArtifactStoreV2.cpp
)

add_library(aitrain_v2_model STATIC
    include/aitrain/v2/ModelManifestV2.h
    include/aitrain/v2/RuntimeAdapterV2.h
    include/aitrain/v2/RuntimeInvocationV2.h
    src/v2/ModelManifestV2.cpp
    src/v2/RuntimeAdapterV2.cpp
    src/v2/RuntimeInvocationV2.cpp
)
target_include_directories(aitrain_v2_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_v2_model PUBLIC aitrain_v2_domain Qt${QT_VERSION_MAJOR}::Core)
target_compile_definitions(aitrain_v2_model PUBLIC QT_NO_CAST_FROM_ASCII)

add_library(aitrain_v2_onnx_runtime STATIC
    include/aitrain/v2/OnnxRuntimeAdapterV2.h
    src/v2/OnnxRuntimeAdapterV2.cpp
)
target_include_directories(aitrain_v2_onnx_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_v2_onnx_runtime PUBLIC aitrain_v2_model aitrain_core Qt${QT_VERSION_MAJOR}::Core)
target_compile_definitions(aitrain_v2_onnx_runtime PUBLIC QT_NO_CAST_FROM_ASCII)

add_library(aitrain_v2_native_runtimes STATIC
    include/aitrain/v2/NcnnRuntimeAdapterV2.h
    include/aitrain/v2/TensorRtRuntimeAdapterV2.h
    src/v2/NcnnRuntimeAdapterV2.cpp
    src/v2/TensorRtRuntimeAdapterV2.cpp
)
target_include_directories(aitrain_v2_native_runtimes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_v2_native_runtimes PUBLIC aitrain_v2_model aitrain_core Qt${QT_VERSION_MAJOR}::Core)
target_compile_definitions(aitrain_v2_native_runtimes PUBLIC QT_NO_CAST_FROM_ASCII)

add_library(aitrain_v2_runtime_matrix STATIC
    include/aitrain/v2/RuntimeCapabilityMatrixV2.h
    src/v2/RuntimeCapabilityMatrixV2.cpp
)
target_include_directories(aitrain_v2_runtime_matrix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_v2_runtime_matrix PUBLIC aitrain_v2_model aitrain_core Qt${QT_VERSION_MAJOR}::Core)
target_compile_definitions(aitrain_v2_runtime_matrix PUBLIC QT_NO_CAST_FROM_ASCII)

target_include_directories(aitrain_v2_artifact
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_v2_artifact
    PUBLIC
        aitrain_v2_storage
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_v2_artifact
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_v2_process STATIC
    include/aitrain/v2/AdapterEventServerV2.h
    include/aitrain/v2/ProcessTreeSupervisor.h
    include/aitrain/v2/PythonAdapterHostV2.h
    src/v2/AdapterEventServerV2.cpp
    src/v2/ProcessTreeSupervisor.cpp
    src/v2/PythonAdapterHostV2.cpp
)

target_include_directories(aitrain_v2_process
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_v2_process
    PUBLIC
        aitrain_v2_protocol
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Network
)

target_compile_definitions(aitrain_v2_process
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_v2_dataset STATIC
    include/aitrain/v2/AnomalyFolderDatasetDriverV2.h
    include/aitrain/v2/BuiltinDatasetDriversV2.h
    include/aitrain/v2/DatasetDriverV2.h
    include/aitrain/v2/DatasetSnapshotV2.h
    include/aitrain/v2/PaddleOcrDetDatasetDriverV2.h
    include/aitrain/v2/PaddleOcrRecDatasetDriverV2.h
    include/aitrain/v2/ValidatedDatasetDriverV2.h
    include/aitrain/v2/YoloDetectionDatasetDriverV2.h
    include/aitrain/v2/YoloObbDatasetDriverV2.h
    include/aitrain/v2/YoloSegmentationDatasetDriverV2.h
    src/v2/AnomalyFolderDatasetDriverV2.cpp
    src/v2/BuiltinDatasetDriversV2.cpp
    src/v2/DatasetDriverV2.cpp
    src/v2/DatasetSnapshotV2.cpp
    src/v2/PaddleOcrDetDatasetDriverV2.cpp
    src/v2/PaddleOcrRecDatasetDriverV2.cpp
    src/v2/ValidatedDatasetDriverV2.cpp
    src/v2/YoloDetectionDatasetDriverV2.cpp
    src/v2/YoloObbDatasetDriverV2.cpp
    src/v2/YoloSegmentationDatasetDriverV2.cpp
)

target_include_directories(aitrain_v2_dataset
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_v2_dataset
    PUBLIC
        aitrain_v2_domain
        aitrain_core
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_v2_dataset
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_core STATIC
    include/aitrain/core/CapabilityRegistry.h
    include/aitrain/core/DatasetValidation.h
    include/aitrain/core/AnnotationIntegration.h
    include/aitrain/core/Cancellation.h
    include/aitrain/core/Deployment.h
    include/aitrain/core/DatasetConversion.h
    include/aitrain/core/DatasetValidators.h
    include/aitrain/core/DetectionDataset.h
    include/aitrain/core/DetectionTrainer.h
    include/aitrain/core/OcrRecDataset.h
    include/aitrain/core/SegmentationDataset.h
    include/aitrain/core/SemanticMask.h
    include/aitrain/core/LicenseManager.h
    include/aitrain/core/LicenseSecurity.h
    include/aitrain/core/VisionModelExport.h
    include/aitrain/core/VisionModelRuntime.h
    include/aitrain/core/VisionPostprocess.h
    include/aitrain/core/WorkerProtocol.h
    include/aitrain/core/WorkflowResult.h
    src/CapabilityRegistry.cpp
    src/AnnotationIntegration.cpp
    src/LicenseManager.cpp
    src/LicenseSecurity.cpp
    src/Deployment.cpp
    src/DatasetConversion.cpp
    src/DatasetConversionCoco.cpp
    src/DatasetConversionCommon.cpp
    src/DatasetConversionInternal.h
    src/DatasetConversionVoc.cpp
    src/DatasetConversionYolo.cpp
    src/DatasetValidators.cpp
    src/DatasetValidation.cpp
    src/YoloDatasetLayout.h
    src/YoloDatasetLayout.cpp
    src/DetectionDataset.cpp
    src/DetectionTrainer.cpp
    src/DetectionTrainerInternal.h
    src/VisionModelExport.cpp
    src/VisionModelRuntimeHelpers.cpp
    src/VisionModelRuntimeNcnn.cpp
    src/VisionModelRuntimeOnnx.cpp
    src/VisionModelRuntimeTensorRt.cpp
    src/VisionPostprocessCommon.cpp
    src/VisionPostprocessOcr.cpp
    src/VisionPostprocessOutput.cpp
    src/VisionPostprocessYolo.cpp
    src/OcrRecDataset.cpp
    src/SegmentationDataset.cpp
    src/SemanticMask.cpp
    src/WorkerProtocol.cpp
)

target_include_directories(aitrain_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_core
    PUBLIC
        aitrain_qt5_msvc_compat
        aitrain_v2_protocol
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Network
        Qt${QT_VERSION_MAJOR}::Sql
        Qt${QT_VERSION_MAJOR}::Xml
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

if(AITRAIN_WITH_NCNN)
    target_compile_definitions(aitrain_core
        PUBLIC
            AITRAIN_WITH_NCNN
    )
    target_include_directories(aitrain_core
        PRIVATE
            ${AITRAIN_NCNN_INCLUDE_DIR}
    )
    target_link_libraries(aitrain_core
        PRIVATE
            ${AITRAIN_NCNN_LIBRARY}
    )
endif()

if(WIN32)
    target_link_libraries(aitrain_core
        PRIVATE
            bcrypt
            crypt32
            advapi32
    )
endif()

if(AITRAIN_WITH_TENSORRT_SDK)
    target_compile_definitions(aitrain_core
        PUBLIC
            AITRAIN_WITH_TENSORRT_SDK
    )
    target_include_directories(aitrain_core
        PRIVATE
            ${AITRAIN_TENSORRT_INCLUDE_DIR}
            ${AITRAIN_CUDA_INCLUDE_DIR}
            ${AITRAIN_CUDA_CRT_INCLUDE_DIR}
    )
endif()
