add_library(aitrain_domain STATIC
    include/aitrain/domain/DomainTypes.h
    src/domain/DomainTypes.cpp
)

target_include_directories(aitrain_domain
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_domain
    PUBLIC
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_domain
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_protocol STATIC
    include/aitrain/protocol/Protocol.h
    include/aitrain/protocol/ProtocolSanitizer.h
    src/protocol/Protocol.cpp
    src/protocol/ProtocolSanitizer.cpp
)

target_include_directories(aitrain_protocol
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_protocol
    PUBLIC
        aitrain_domain
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_protocol
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_storage STATIC
    include/aitrain/storage/ProjectStore.h
    src/storage/ProjectStore.cpp
)

add_library(aitrain_training_workflow_profile STATIC
    include/aitrain/workflow/TrainingWorkflowProfile.h
    src/workflow/TrainingWorkflowProfile.cpp
)

target_include_directories(aitrain_training_workflow_profile
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_training_workflow_profile
    PUBLIC
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_training_workflow_profile
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

target_include_directories(aitrain_storage
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_storage
    PUBLIC
        aitrain_domain
        aitrain_model
        aitrain_protocol
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql
)

target_compile_definitions(aitrain_storage
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_application STATIC
    include/aitrain/workflow/CapabilityPlanner.h
    include/aitrain/workflow/TaskCoordinator.h
    include/aitrain/workflow/TaskExecutionHost.h
    include/aitrain/workflow/WorkflowRunner.h
    include/aitrain/workflow/EvidenceBundle.h
    include/aitrain/workflow/EvidenceRenderer.h
    include/aitrain/model/ModelImportService.h
    include/aitrain/runtime/ModelPackageRuntimeService.h
    include/aitrain/workflow/ProjectWorkspace.h
    include/aitrain/workflow/ProjectQueryService.h
    src/workflow/CapabilityPlanner.cpp
    src/workflow/TaskCoordinator.cpp
    src/workflow/TaskExecutionHost.cpp
    src/workflow/WorkflowRunner.cpp
    src/workflow/EvidenceBundle.cpp
    src/workflow/EvidenceRenderer.cpp
    src/model/ModelImportService.cpp
    src/runtime/ModelPackageRuntimeService.cpp
    src/workflow/ProjectWorkspace.cpp
    src/workflow/ProjectWorkspaceAnnotation.cpp
    src/workflow/ProjectWorkspaceDataQuality.cpp
    src/workflow/ProjectWorkspaceDiagnostics.cpp
    src/workflow/ProjectWorkspaceExternalEvidence.cpp
    src/workflow/ProjectWorkspaceDatasetConversion.cpp
    src/workflow/ProjectWorkspaceDatasetSnapshotImport.cpp
    src/workflow/ProjectWorkspaceDatasetSplit.cpp
    src/workflow/ProjectWorkspaceOcrAcceptance.cpp
    src/workflow/ProjectWorkspaceOcrReportPackager.cpp
    src/workflow/ProjectWorkspaceRuntimeDelivery.cpp
    src/workflow/ProjectQueryService.cpp
)

add_library(aitrain_dataset_conversion STATIC
    include/aitrain/dataset/DatasetConversionService.h
    src/dataset/DatasetConversionService.cpp
)

target_include_directories(aitrain_dataset_conversion PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_dataset_conversion
    PUBLIC
        aitrain_artifact
        aitrain_dataset
        aitrain_foundation
        Qt${QT_VERSION_MAJOR}::Core
)
target_compile_definitions(aitrain_dataset_conversion PUBLIC QT_NO_CAST_FROM_ASCII)

target_include_directories(aitrain_application
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_application
    PUBLIC
        aitrain_training_workflow_profile
        aitrain_protocol
        aitrain_process
        aitrain_storage
        aitrain_artifact
        aitrain_runtime_matrix
        aitrain_onnx_runtime
        aitrain_native_runtimes
        aitrain_dataset
        aitrain_dataset_conversion
        aitrain_foundation
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_application
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_artifact STATIC
    include/aitrain/artifact/ArtifactStore.h
    src/artifact/ArtifactStore.cpp
)

add_library(aitrain_model STATIC
    include/aitrain/model/ModelManifest.h
    include/aitrain/runtime/RuntimeAdapter.h
    include/aitrain/runtime/RuntimeInvocation.h
    src/model/ModelManifest.cpp
    src/runtime/RuntimeAdapter.cpp
    src/runtime/RuntimeInvocation.cpp
)
target_include_directories(aitrain_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_model PUBLIC aitrain_domain Qt${QT_VERSION_MAJOR}::Core)
target_compile_definitions(aitrain_model PUBLIC QT_NO_CAST_FROM_ASCII)

add_library(aitrain_onnx_runtime STATIC
    include/aitrain/runtime/OnnxRuntimeAdapter.h
    src/runtime/OnnxRuntimeAdapter.cpp
)
target_include_directories(aitrain_onnx_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_onnx_runtime PUBLIC aitrain_model aitrain_foundation Qt${QT_VERSION_MAJOR}::Core)
target_compile_definitions(aitrain_onnx_runtime PUBLIC QT_NO_CAST_FROM_ASCII)

add_library(aitrain_native_runtimes STATIC
    include/aitrain/runtime/NcnnRuntimeAdapter.h
    include/aitrain/runtime/TensorRtRuntimeAdapter.h
    src/runtime/NcnnRuntimeAdapter.cpp
    src/runtime/TensorRtRuntimeAdapter.cpp
)
target_include_directories(aitrain_native_runtimes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_native_runtimes PUBLIC aitrain_model aitrain_foundation Qt${QT_VERSION_MAJOR}::Core)
target_compile_definitions(aitrain_native_runtimes PUBLIC QT_NO_CAST_FROM_ASCII)

add_library(aitrain_runtime_matrix STATIC
    include/aitrain/runtime/RuntimeCapabilityMatrix.h
    src/runtime/RuntimeCapabilityMatrix.cpp
)
target_include_directories(aitrain_runtime_matrix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_runtime_matrix PUBLIC aitrain_model aitrain_foundation Qt${QT_VERSION_MAJOR}::Core)
target_compile_definitions(aitrain_runtime_matrix PUBLIC QT_NO_CAST_FROM_ASCII)

target_include_directories(aitrain_artifact
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_artifact
    PUBLIC
        aitrain_storage
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_artifact
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_process STATIC
    include/aitrain/process/AdapterEventServer.h
    include/aitrain/process/ProcessTreeSupervisor.h
    include/aitrain/process/PythonAdapterHost.h
    src/process/AdapterEventServer.cpp
    src/process/ProcessTreeSupervisor.cpp
    src/process/PythonAdapterHost.cpp
)

target_include_directories(aitrain_process
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_process
    PUBLIC
        aitrain_protocol
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Network
)

target_compile_definitions(aitrain_process
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_dataset STATIC
    include/aitrain/dataset/AnomalyFolderDatasetDriver.h
    include/aitrain/dataset/BuiltinDatasetDrivers.h
    include/aitrain/dataset/DatasetDriver.h
    include/aitrain/dataset/DatasetSnapshot.h
    include/aitrain/dataset/PaddleOcrDetDatasetDriver.h
    include/aitrain/dataset/PaddleOcrRecDatasetDriver.h
    include/aitrain/dataset/ValidatedDatasetDriver.h
    include/aitrain/dataset/YoloDetectionDatasetDriver.h
    include/aitrain/dataset/YoloObbDatasetDriver.h
    include/aitrain/dataset/YoloSegmentationDatasetDriver.h
    src/dataset/AnomalyFolderDatasetDriver.cpp
    src/dataset/BuiltinDatasetDrivers.cpp
    src/dataset/DatasetDriver.cpp
    src/dataset/DatasetSnapshot.cpp
    src/dataset/PaddleOcrDetDatasetDriver.cpp
    src/dataset/PaddleOcrRecDatasetDriver.cpp
    src/dataset/ValidatedDatasetDriver.cpp
    src/dataset/YoloDetectionDatasetDriver.cpp
    src/dataset/YoloObbDatasetDriver.cpp
    src/dataset/YoloSegmentationDatasetDriver.cpp
)

target_include_directories(aitrain_dataset
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_dataset
    PUBLIC
        aitrain_domain
        aitrain_foundation
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_dataset
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_foundation STATIC
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
    include/aitrain/core/TaskTypes.h
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

target_include_directories(aitrain_foundation
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_foundation
    PUBLIC
        aitrain_qt5_msvc_compat
        aitrain_protocol
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Network
        Qt${QT_VERSION_MAJOR}::Sql
        Qt${QT_VERSION_MAJOR}::Xml
)

target_compile_definitions(aitrain_foundation
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

if(AITRAIN_WITH_ONNXRUNTIME)
    target_compile_definitions(aitrain_foundation
        PUBLIC
            AITRAIN_WITH_ONNXRUNTIME
    )
    target_include_directories(aitrain_foundation
        PRIVATE
            ${AITRAIN_ONNXRUNTIME_INCLUDE_DIR}
    )
    target_link_libraries(aitrain_foundation
        PRIVATE
            ${AITRAIN_ONNXRUNTIME_LIBRARY}
    )
endif()

if(AITRAIN_WITH_NCNN)
    target_compile_definitions(aitrain_foundation
        PUBLIC
            AITRAIN_WITH_NCNN
    )
    target_include_directories(aitrain_foundation
        PRIVATE
            ${AITRAIN_NCNN_INCLUDE_DIR}
    )
    target_link_libraries(aitrain_foundation
        PRIVATE
            ${AITRAIN_NCNN_LIBRARY}
    )
endif()

if(WIN32)
    target_link_libraries(aitrain_foundation
        PRIVATE
            bcrypt
            crypt32
            advapi32
    )
endif()

if(AITRAIN_WITH_TENSORRT_SDK)
    target_compile_definitions(aitrain_foundation
        PUBLIC
            AITRAIN_WITH_TENSORRT_SDK
    )
    target_include_directories(aitrain_foundation
        PRIVATE
            ${AITRAIN_TENSORRT_INCLUDE_DIR}
            ${AITRAIN_CUDA_INCLUDE_DIR}
            ${AITRAIN_CUDA_CRT_INCLUDE_DIR}
    )
endif()
