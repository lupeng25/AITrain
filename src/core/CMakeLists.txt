add_library(aitrain_domain STATIC
    include/aitrain/domain/DomainTypes.h
    include/aitrain/domain/Pagination.h
    include/aitrain/domain/ArtifactMemberPath.h
    src/domain/DomainTypes.cpp
    src/domain/ArtifactMemberPath.cpp
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
    include/aitrain/storage/ProjectDatabase.h
    include/aitrain/storage/ProjectStore.h
    src/storage/ProjectDatabase.cpp
    src/storage/ProjectStore.cpp
)

add_library(aitrain_training_workflow_profile STATIC
    include/aitrain/workflow/TrainingWorkflowProfile.h
    src/workflow/TrainingWorkflowProfile.cpp
)

add_library(aitrain_product_contract STATIC
    include/aitrain/product/ProductCapabilityContract.h
    src/product/ProductCapabilityContract.cpp
)

target_include_directories(aitrain_product_contract
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_product_contract
    PUBLIC
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_product_contract
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

target_include_directories(aitrain_training_workflow_profile
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_training_workflow_profile
    PUBLIC
        aitrain_product_contract
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_training_workflow_profile
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

# 这三个边界原先被历史 foundation 聚合库携带。它们没有依赖训练/运行时
# 实现，单独成 target 后，GUI、Worker 和许可证工具可以按实际能力声明依赖。
add_library(aitrain_capability_registry STATIC
    include/aitrain/core/CapabilityRegistry.h
    src/CapabilityRegistry.cpp
)
target_include_directories(aitrain_capability_registry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_capability_registry PUBLIC aitrain_product_contract Qt${QT_VERSION_MAJOR}::Core)
target_compile_definitions(aitrain_capability_registry PUBLIC QT_NO_CAST_FROM_ASCII)

add_library(aitrain_license STATIC
    include/aitrain/core/LicenseManager.h
    include/aitrain/core/LicenseSecurity.h
    src/LicenseManager.cpp
    src/LicenseSecurity.cpp
)
target_include_directories(aitrain_license PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_license PUBLIC Qt${QT_VERSION_MAJOR}::Core)
target_compile_definitions(aitrain_license PUBLIC QT_NO_CAST_FROM_ASCII)
if(WIN32)
    target_link_libraries(aitrain_license PRIVATE bcrypt crypt32 advapi32)
endif()

add_library(aitrain_worker_protocol STATIC
    include/aitrain/core/Cancellation.h
    include/aitrain/core/TaskTypes.h
    include/aitrain/core/WorkerProtocol.h
    include/aitrain/core/WorkflowResult.h
    src/WorkerProtocol.cpp
)
target_include_directories(aitrain_worker_protocol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_worker_protocol PUBLIC aitrain_domain aitrain_protocol Qt${QT_VERSION_MAJOR}::Core)
target_compile_definitions(aitrain_worker_protocol PUBLIC QT_NO_CAST_FROM_ASCII)

# 数据集校验只依赖 Qt 的值类型和 YAML/文件处理辅助，不依赖 Workflow、模型
# 运行时或存储。将它从历史 foundation 聚合库抽出，避免所有消费者被迫链接
# 视觉/训练实现，同时让 dataset driver/conversion 共享同一稳定边界。
add_library(aitrain_dataset_validation STATIC
    include/aitrain/core/DatasetValidation.h
    include/aitrain/core/DatasetValidators.h
    include/aitrain/core/SemanticMask.h
    src/DatasetValidation.cpp
    src/DatasetValidators.cpp
    src/SemanticMask.cpp
    src/YoloDatasetLayout.h
    src/YoloDatasetLayout.cpp
)
target_include_directories(aitrain_dataset_validation
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(aitrain_dataset_validation
    PUBLIC
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
)
target_compile_definitions(aitrain_dataset_validation PUBLIC QT_NO_CAST_FROM_ASCII)

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
    include/aitrain/core/DatasetConversion.h
    src/dataset/DatasetConversionService.cpp
    src/DatasetConversion.cpp
    src/DatasetConversionCoco.cpp
    src/DatasetConversionCommon.cpp
    src/DatasetConversionInternal.h
    src/DatasetConversionVoc.cpp
    src/DatasetConversionYolo.cpp
)

target_include_directories(aitrain_dataset_conversion PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_dataset_conversion
    PUBLIC
        aitrain_artifact
        aitrain_annotation_integration
        aitrain_dataset
        aitrain_dataset_validation
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Xml
)
target_compile_definitions(aitrain_dataset_conversion PUBLIC QT_NO_CAST_FROM_ASCII)

target_include_directories(aitrain_application
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aitrain_application
    PUBLIC
        aitrain_capability_registry
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
        aitrain_vision_data
        aitrain_annotation_integration
        aitrain_model_export
        aitrain_vision_runtime
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_application
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_artifact STATIC
    include/aitrain/artifact/ArtifactStore.h
    include/aitrain/artifact/VerifiedArtifactReader.h
    src/artifact/ArtifactStore.cpp
    src/artifact/VerifiedArtifactReader.cpp
)

add_library(aitrain_model STATIC
    include/aitrain/model/ModelManifest.h
    include/aitrain/runtime/RuntimeAdapter.h
    include/aitrain/runtime/RuntimeInvocation.h
    include/aitrain/runtime/RuntimeBenchmarkRunner.h
    src/model/ModelManifest.cpp
    src/runtime/RuntimeAdapter.cpp
    src/runtime/RuntimeInvocation.cpp
    src/runtime/RuntimeBenchmarkRunner.cpp
)
target_include_directories(aitrain_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_model PUBLIC aitrain_domain Qt${QT_VERSION_MAJOR}::Core)
target_compile_definitions(aitrain_model PUBLIC QT_NO_CAST_FROM_ASCII)

add_library(aitrain_onnx_runtime STATIC
    include/aitrain/runtime/OnnxRuntimeAdapter.h
    src/runtime/OnnxRuntimeAdapter.cpp
)
target_include_directories(aitrain_onnx_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_onnx_runtime PUBLIC aitrain_model aitrain_vision_runtime Qt${QT_VERSION_MAJOR}::Core)
target_compile_definitions(aitrain_onnx_runtime PUBLIC QT_NO_CAST_FROM_ASCII)

add_library(aitrain_native_runtimes STATIC
    include/aitrain/runtime/NcnnRuntimeAdapter.h
    include/aitrain/runtime/TensorRtRuntimeAdapter.h
    src/runtime/NcnnRuntimeAdapter.cpp
    src/runtime/TensorRtRuntimeAdapter.cpp
)
target_include_directories(aitrain_native_runtimes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_native_runtimes PUBLIC aitrain_model aitrain_vision_runtime Qt${QT_VERSION_MAJOR}::Core)
target_compile_definitions(aitrain_native_runtimes PUBLIC QT_NO_CAST_FROM_ASCII)

add_library(aitrain_runtime_matrix STATIC
    include/aitrain/runtime/RuntimeCapabilityMatrix.h
    src/runtime/RuntimeCapabilityMatrix.cpp
)
target_include_directories(aitrain_runtime_matrix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aitrain_runtime_matrix PUBLIC aitrain_model aitrain_product_contract aitrain_vision_runtime Qt${QT_VERSION_MAJOR}::Core)
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
        aitrain_dataset_validation
        Qt${QT_VERSION_MAJOR}::Core
)

target_compile_definitions(aitrain_dataset
    PUBLIC
        QT_NO_CAST_FROM_ASCII
)

add_library(aitrain_vision_data STATIC
    include/aitrain/core/DetectionDataset.h
    include/aitrain/core/OcrRecDataset.h
    include/aitrain/core/SegmentationDataset.h
    src/DetectionDataset.cpp
    src/OcrRecDataset.cpp
    src/SegmentationDataset.cpp
)

add_library(aitrain_annotation_integration STATIC
    include/aitrain/core/AnnotationIntegration.h
    src/AnnotationIntegration.cpp
)

add_library(aitrain_vision_runtime STATIC
    include/aitrain/core/Deployment.h
    include/aitrain/core/TaskTypes.h
    include/aitrain/core/VisionModelRuntime.h
    include/aitrain/core/VisionPostprocess.h
    src/Deployment.cpp
    src/VisionRuntimeShared.cpp
    src/VisionRuntimeInternal.h
    src/VisionModelRuntimeHelpers.cpp
    src/VisionModelRuntimeNcnn.cpp
    src/VisionModelRuntimeOnnx.cpp
    src/VisionModelRuntimeTensorRt.cpp
    src/VisionPostprocessCommon.cpp
    src/VisionPostprocessOcr.cpp
    src/VisionPostprocessOutput.cpp
    src/VisionPostprocessYolo.cpp
)

add_library(aitrain_model_export STATIC
    include/aitrain/core/VisionModelExport.h
    src/VisionModelExport.cpp
)

foreach(target_name IN ITEMS
        aitrain_vision_data
        aitrain_annotation_integration
        aitrain_vision_runtime
        aitrain_model_export)
    target_include_directories(${target_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(${target_name} PUBLIC QT_NO_CAST_FROM_ASCII)
endforeach()

target_link_libraries(aitrain_vision_data PUBLIC
    aitrain_dataset_validation
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui)

target_link_libraries(aitrain_annotation_integration PUBLIC
    aitrain_dataset_validation
    Qt${QT_VERSION_MAJOR}::Core)

target_link_libraries(aitrain_vision_runtime PUBLIC
        aitrain_qt5_msvc_compat
        aitrain_dataset_validation
        aitrain_protocol
        aitrain_vision_data
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Network
        Qt${QT_VERSION_MAJOR}::Sql
        Qt${QT_VERSION_MAJOR}::Xml
)

target_link_libraries(aitrain_model_export PUBLIC
    aitrain_vision_runtime
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui)

if(AITRAIN_WITH_ONNXRUNTIME)
    target_compile_definitions(aitrain_vision_runtime
        PUBLIC
            AITRAIN_WITH_ONNXRUNTIME
    )
    target_include_directories(aitrain_vision_runtime
        PRIVATE
            ${AITRAIN_ONNXRUNTIME_INCLUDE_DIR}
    )
    target_link_libraries(aitrain_vision_runtime
        PRIVATE
            ${AITRAIN_ONNXRUNTIME_LIBRARY}
    )
endif()

if(AITRAIN_WITH_NCNN)
    target_compile_definitions(aitrain_vision_runtime
        PUBLIC
            AITRAIN_WITH_NCNN
    )
    target_include_directories(aitrain_vision_runtime
        PRIVATE
            ${AITRAIN_NCNN_INCLUDE_DIR}
    )
    target_link_libraries(aitrain_vision_runtime
        PRIVATE
            ${AITRAIN_NCNN_LIBRARY}
    )
endif()

if(WIN32)
    target_link_libraries(aitrain_vision_runtime
        PRIVATE
            bcrypt
            crypt32
            advapi32
    )
endif()

if(AITRAIN_WITH_TENSORRT_SDK)
    target_compile_definitions(aitrain_vision_runtime
        PUBLIC
            AITRAIN_WITH_TENSORRT_SDK
    )
    target_include_directories(aitrain_vision_runtime
        PRIVATE
            ${AITRAIN_TENSORRT_INCLUDE_DIR}
            ${AITRAIN_CUDA_INCLUDE_DIR}
            ${AITRAIN_CUDA_CRT_INCLUDE_DIR}
    )
    target_include_directories(aitrain_model_export
        PRIVATE
            ${AITRAIN_TENSORRT_INCLUDE_DIR}
            ${AITRAIN_CUDA_INCLUDE_DIR}
            ${AITRAIN_CUDA_CRT_INCLUDE_DIR}
    )
endif()
