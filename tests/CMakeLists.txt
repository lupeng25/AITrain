get_target_property(AITRAIN_QT_QMAKE_EXECUTABLE
    Qt${QT_VERSION_MAJOR}::qmake IMPORTED_LOCATION)
get_filename_component(AITRAIN_QT_BIN_DIR
    "${AITRAIN_QT_QMAKE_EXECUTABLE}" DIRECTORY)
get_filename_component(AITRAIN_QT_INSTALL_ROOT
    "${AITRAIN_QT_BIN_DIR}" DIRECTORY)
set(AITRAIN_QT_PLUGIN_DIR "${AITRAIN_QT_INSTALL_ROOT}/plugins")
set(AITRAIN_QT_PLATFORM_PLUGIN_DIR "${AITRAIN_QT_PLUGIN_DIR}/platforms")

if(NOT EXISTS "${AITRAIN_QT_PLATFORM_PLUGIN_DIR}")
    message(FATAL_ERROR
        "Qt 平台插件目录不存在: ${AITRAIN_QT_PLATFORM_PLUGIN_DIR}")
endif()

set(AITRAIN_QT_TEST_ENVIRONMENT
    "QT_PLUGIN_PATH=${AITRAIN_QT_PLUGIN_DIR}"
    "QT_QPA_PLATFORM_PLUGIN_PATH=${AITRAIN_QT_PLATFORM_PLUGIN_DIR}"
    "QT_QPA_PLATFORM=offscreen"
)

function(add_aitrain_qt_test target source)
    add_executable(${target}
        ../src/app/src/DatasetConversionUiModel.h
        ../src/app/src/DatasetConversionUiModel.cpp
        ../src/app/src/WorkerClient.h
        ../src/app/src/WorkerClient.cpp
        TestSupport.h
        TestSupport.cpp
        ${source}
    )

    target_include_directories(${target}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/app/src
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(${target}
        PRIVATE
            aitrain_core
            Qt${QT_VERSION_MAJOR}::Core
            Qt${QT_VERSION_MAJOR}::Network
            Qt${QT_VERSION_MAJOR}::Test
            Qt${QT_VERSION_MAJOR}::Sql
    )

    if(AITRAIN_WITH_TENSORRT_SDK)
        target_include_directories(${target}
            PRIVATE
                ${AITRAIN_TENSORRT_INCLUDE_DIR}
                ${AITRAIN_CUDA_INCLUDE_DIR}
                ${AITRAIN_CUDA_CRT_INCLUDE_DIR}
        )
    endif()
    if(AITRAIN_WITH_NCNN)
        target_include_directories(${target}
            PRIVATE
                ${AITRAIN_NCNN_INCLUDE_DIR}
        )
    endif()

    set(test_output "${CMAKE_BINARY_DIR}/tests/${target}_ctest.txt")
    add_test(NAME ${target}
        COMMAND ${target} -o "${test_output},txt"
    )
    set_tests_properties(${target} PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        ATTACHED_FILES_ON_FAIL "${test_output}"
        ENVIRONMENT "${AITRAIN_QT_TEST_ENVIRONMENT}"
    )

    aitrain_copy_onnxruntime_runtime(${target})
    aitrain_copy_ncnn_runtime(${target})
    aitrain_copy_tensorrt_runtime(${target})
    aitrain_copy_qt_platform_runtime(${target})
endfunction()

add_aitrain_qt_test(aitrain_platform_tests tst_platform.cpp)
target_link_libraries(aitrain_platform_tests PRIVATE aitrain_v2_protocol)
add_aitrain_qt_test(aitrain_dataset_tests tst_datasets.cpp)
add_aitrain_qt_test(aitrain_dataset_conversion_tests tst_dataset_conversion.cpp)
add_aitrain_qt_test(aitrain_dataset_conversion_ui_tests tst_dataset_conversion_ui.cpp)
add_aitrain_qt_test(aitrain_annotation_integration_tests tst_annotation_integration.cpp)
if(WIN32)
    add_aitrain_qt_test(aitrain_license_security_tests tst_license_security.cpp)
endif()
add_aitrain_qt_test(aitrain_detection_inference_tests tst_detection_inference.cpp)
add_aitrain_qt_test(aitrain_ocr_segmentation_worker_tests tst_ocr_segmentation_worker.cpp)
target_link_libraries(aitrain_ocr_segmentation_worker_tests PRIVATE
    aitrain_v2_application aitrain_v2_storage aitrain_v2_protocol)

add_executable(aitrain_v2_domain_tests
    tst_v2_domain.cpp
)

target_link_libraries(aitrain_v2_domain_tests
    PRIVATE
        aitrain_v2_domain
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Test
)

set(v2_domain_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_domain_tests_ctest.txt")
add_test(NAME aitrain_v2_domain_tests
    COMMAND aitrain_v2_domain_tests -o "${v2_domain_test_output},txt"
)
set_tests_properties(aitrain_v2_domain_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_domain_test_output}"
)

add_executable(aitrain_v2_training_workflow_profile_tests
    tst_v2_training_workflow_profile.cpp
)

target_link_libraries(aitrain_v2_training_workflow_profile_tests
    PRIVATE
        aitrain_v2_training_workflow_profile
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Test
)

set(v2_training_workflow_profile_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_training_workflow_profile_tests_ctest.txt")
add_test(NAME aitrain_v2_training_workflow_profile_tests
    COMMAND aitrain_v2_training_workflow_profile_tests -o "${v2_training_workflow_profile_test_output},txt"
)
set_tests_properties(aitrain_v2_training_workflow_profile_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_training_workflow_profile_test_output}"
)

add_executable(aitrain_v2_protocol_tests
    tst_v2_protocol.cpp
)

target_link_libraries(aitrain_v2_protocol_tests
    PRIVATE
        aitrain_v2_protocol
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Test
)

set(v2_protocol_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_protocol_tests_ctest.txt")
add_test(NAME aitrain_v2_protocol_tests
    COMMAND aitrain_v2_protocol_tests -o "${v2_protocol_test_output},txt"
)
set_tests_properties(aitrain_v2_protocol_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_protocol_test_output}"
)

add_executable(aitrain_v2_storage_tests
    tst_v2_storage.cpp
)

target_link_libraries(aitrain_v2_storage_tests
    PRIVATE
        aitrain_v2_storage
        aitrain_v2_model
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql
        Qt${QT_VERSION_MAJOR}::Test
)

set(v2_storage_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_storage_tests_ctest.txt")
add_test(NAME aitrain_v2_storage_tests
    COMMAND aitrain_v2_storage_tests -o "${v2_storage_test_output},txt"
)
set_tests_properties(aitrain_v2_storage_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_storage_test_output}"
)

add_executable(aitrain_v2_application_tests
    tst_v2_application.cpp
)

add_executable(aitrain_diagnostics_v2_tests
    ../src/app/src/DiagnosticBundlePresenterV2.h
    ../src/app/src/DiagnosticBundlePresenterV2.cpp
    ../src/app/src/EnvironmentCheckPresenterV2.h
    ../src/app/src/EnvironmentCheckPresenterV2.cpp
    tst_diagnostics_v2.cpp
)
target_include_directories(aitrain_diagnostics_v2_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/app/src)
target_link_libraries(aitrain_diagnostics_v2_tests PRIVATE
    aitrain_v2_application aitrain_core Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Sql Qt${QT_VERSION_MAJOR}::Test)
set(diagnostics_v2_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_diagnostics_v2_tests_ctest.txt")
add_test(NAME aitrain_diagnostics_v2_tests
    COMMAND aitrain_diagnostics_v2_tests -o "${diagnostics_v2_test_output},txt")
set_tests_properties(aitrain_diagnostics_v2_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${diagnostics_v2_test_output}")

target_link_libraries(aitrain_v2_application_tests
    PRIVATE
        aitrain_v2_application
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql
        Qt${QT_VERSION_MAJOR}::Test
)

set(v2_application_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_application_tests_ctest.txt")
add_test(NAME aitrain_v2_application_tests
    COMMAND aitrain_v2_application_tests -o "${v2_application_test_output},txt"
)
set_tests_properties(aitrain_v2_application_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_application_test_output}"
)

add_executable(aitrain_v2_project_summary_query_tests
    tst_v2_project_summary_query.cpp
)
target_link_libraries(aitrain_v2_project_summary_query_tests
    PRIVATE
        aitrain_v2_application
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql
        Qt${QT_VERSION_MAJOR}::Test
)
set(v2_project_summary_query_test_output
    "${CMAKE_BINARY_DIR}/tests/aitrain_v2_project_summary_query_tests_ctest.txt")
add_test(NAME aitrain_v2_project_summary_query_tests
    COMMAND aitrain_v2_project_summary_query_tests
        -o "${v2_project_summary_query_test_output},txt"
)
set_tests_properties(aitrain_v2_project_summary_query_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_project_summary_query_test_output}"
)

add_executable(aitrain_task_artifact_presenter_v2_tests
    ../src/app/src/TaskArtifactPresenterV2.h
    ../src/app/src/TaskArtifactPresenterV2.cpp
    tst_task_artifact_presenter_v2.cpp
)
target_include_directories(aitrain_task_artifact_presenter_v2_tests
    PRIVATE ${CMAKE_SOURCE_DIR}/src/app/src)
target_link_libraries(aitrain_task_artifact_presenter_v2_tests
    PRIVATE aitrain_v2_application Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql Qt${QT_VERSION_MAJOR}::Test)
set(task_artifact_presenter_v2_test_output
    "${CMAKE_BINARY_DIR}/tests/aitrain_task_artifact_presenter_v2_tests_ctest.txt")
add_test(NAME aitrain_task_artifact_presenter_v2_tests
    COMMAND aitrain_task_artifact_presenter_v2_tests
        -o "${task_artifact_presenter_v2_test_output},txt")
set_tests_properties(aitrain_task_artifact_presenter_v2_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${task_artifact_presenter_v2_test_output}")

add_executable(aitrain_project_summary_presenter_v2_tests
    ../src/app/src/ProjectSummaryPresenterV2.h
    ../src/app/src/ProjectSummaryPresenterV2.cpp
    tst_project_summary_presenter_v2.cpp
)
target_include_directories(aitrain_project_summary_presenter_v2_tests
    PRIVATE ${CMAKE_SOURCE_DIR}/src/app/src)
target_link_libraries(aitrain_project_summary_presenter_v2_tests
    PRIVATE aitrain_v2_application Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql Qt${QT_VERSION_MAJOR}::Test)
set(project_summary_presenter_v2_test_output
    "${CMAKE_BINARY_DIR}/tests/aitrain_project_summary_presenter_v2_tests_ctest.txt")
add_test(NAME aitrain_project_summary_presenter_v2_tests
    COMMAND aitrain_project_summary_presenter_v2_tests
        -o "${project_summary_presenter_v2_test_output},txt")
set_tests_properties(aitrain_project_summary_presenter_v2_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${project_summary_presenter_v2_test_output}")

add_executable(aitrain_model_registry_presenter_v2_tests
    ../src/app/src/ModelRegistryPresenterV2.h
    ../src/app/src/ModelRegistryPresenterV2.cpp
    tst_model_registry_presenter_v2.cpp
)
target_include_directories(aitrain_model_registry_presenter_v2_tests
    PRIVATE ${CMAKE_SOURCE_DIR}/src/app/src)
target_link_libraries(aitrain_model_registry_presenter_v2_tests
    PRIVATE aitrain_v2_application Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql Qt${QT_VERSION_MAJOR}::Test)
set(model_registry_presenter_v2_test_output
    "${CMAKE_BINARY_DIR}/tests/aitrain_model_registry_presenter_v2_tests_ctest.txt")
add_test(NAME aitrain_model_registry_presenter_v2_tests
    COMMAND aitrain_model_registry_presenter_v2_tests
        -o "${model_registry_presenter_v2_test_output},txt")
set_tests_properties(aitrain_model_registry_presenter_v2_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${model_registry_presenter_v2_test_output}")

add_executable(aitrain_v2_evidence_tests
    tst_v2_evidence.cpp
)

target_link_libraries(aitrain_v2_evidence_tests
    PRIVATE
        aitrain_v2_application
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Test
)

set(v2_evidence_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_evidence_tests_ctest.txt")
add_test(NAME aitrain_v2_evidence_tests
    COMMAND aitrain_v2_evidence_tests -o "${v2_evidence_test_output},txt"
)
set_tests_properties(aitrain_v2_evidence_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_evidence_test_output}"
)

add_executable(aitrain_v2_artifact_tests
    tst_v2_artifact_store.cpp
)

add_executable(aitrain_v2_model_manifest_tests tst_v2_model_manifest.cpp)
target_link_libraries(aitrain_v2_model_manifest_tests PRIVATE aitrain_v2_model aitrain_v2_onnx_runtime aitrain_v2_runtime_matrix Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Test)
aitrain_copy_onnxruntime_runtime(aitrain_v2_model_manifest_tests)
set(v2_model_manifest_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_model_manifest_tests_ctest.txt")
add_test(NAME aitrain_v2_model_manifest_tests COMMAND aitrain_v2_model_manifest_tests -o "${v2_model_manifest_test_output},txt")
set_tests_properties(aitrain_v2_model_manifest_tests PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" ATTACHED_FILES_ON_FAIL "${v2_model_manifest_test_output}")

add_executable(aitrain_v2_runtime_adapter_tests tst_v2_runtime_adapters.cpp)
target_link_libraries(aitrain_v2_runtime_adapter_tests PRIVATE aitrain_v2_native_runtimes aitrain_v2_runtime_matrix Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Test)
set(v2_runtime_adapter_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_runtime_adapter_tests_ctest.txt")
add_test(NAME aitrain_v2_runtime_adapter_tests COMMAND aitrain_v2_runtime_adapter_tests -o "${v2_runtime_adapter_test_output},txt")
set_tests_properties(aitrain_v2_runtime_adapter_tests PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" ATTACHED_FILES_ON_FAIL "${v2_runtime_adapter_test_output}")

add_executable(aitrain_v2_runtime_delivery_workflow_tests
    tst_v2_runtime_delivery_workflow.cpp)
target_link_libraries(aitrain_v2_runtime_delivery_workflow_tests
    PRIVATE
        aitrain_v2_application
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Sql
        Qt${QT_VERSION_MAJOR}::Test)
set(v2_runtime_delivery_workflow_test_output
    "${CMAKE_BINARY_DIR}/tests/aitrain_v2_runtime_delivery_workflow_tests_ctest.txt")
add_test(NAME aitrain_v2_runtime_delivery_workflow_tests
    COMMAND aitrain_v2_runtime_delivery_workflow_tests
        -o "${v2_runtime_delivery_workflow_test_output},txt")
set_tests_properties(aitrain_v2_runtime_delivery_workflow_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_runtime_delivery_workflow_test_output}")

target_link_libraries(aitrain_v2_artifact_tests
    PRIVATE
        aitrain_v2_artifact
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql
        Qt${QT_VERSION_MAJOR}::Test
)

set(v2_artifact_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_artifact_tests_ctest.txt")
add_test(NAME aitrain_v2_artifact_tests
    COMMAND aitrain_v2_artifact_tests -o "${v2_artifact_test_output},txt"
)
set_tests_properties(aitrain_v2_artifact_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_artifact_test_output}"
)

add_executable(aitrain_v2_process_tests
    tst_v2_process_tree.cpp
)

target_link_libraries(aitrain_v2_process_tests
    PRIVATE
        aitrain_v2_process
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Network
        Qt${QT_VERSION_MAJOR}::Test
)

set(v2_process_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_process_tests_ctest.txt")
add_test(NAME aitrain_v2_process_tests
    COMMAND aitrain_v2_process_tests -o "${v2_process_test_output},txt"
)
set_tests_properties(aitrain_v2_process_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_process_test_output}"
)

add_executable(aitrain_v2_dataset_tests
    tst_v2_dataset_snapshot.cpp
)

target_link_libraries(aitrain_v2_dataset_tests
    PRIVATE
        aitrain_v2_dataset
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Test
)

set(v2_dataset_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_dataset_tests_ctest.txt")
add_test(NAME aitrain_v2_dataset_tests
    COMMAND aitrain_v2_dataset_tests -o "${v2_dataset_test_output},txt"
)
set_tests_properties(aitrain_v2_dataset_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_dataset_test_output}"
)

add_executable(aitrain_v2_dataset_conversion_tests
    tst_v2_dataset_conversion.cpp
)

target_link_libraries(aitrain_v2_dataset_conversion_tests
    PRIVATE
        aitrain_v2_dataset_conversion
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Test
)

set(v2_dataset_conversion_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_dataset_conversion_tests_ctest.txt")
add_test(NAME aitrain_v2_dataset_conversion_tests
    COMMAND aitrain_v2_dataset_conversion_tests -o "${v2_dataset_conversion_test_output},txt"
)
set_tests_properties(aitrain_v2_dataset_conversion_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_dataset_conversion_test_output}"
)

add_executable(aitrain_v2_data_quality_tests
    tst_v2_data_quality.cpp
)
target_link_libraries(aitrain_v2_data_quality_tests
    PRIVATE
        aitrain_v2_application
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Sql
        Qt${QT_VERSION_MAJOR}::Test
)
set(v2_data_quality_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_data_quality_tests_ctest.txt")
add_test(NAME aitrain_v2_data_quality_tests
    COMMAND aitrain_v2_data_quality_tests -o "${v2_data_quality_test_output},txt"
)
set_tests_properties(aitrain_v2_data_quality_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_data_quality_test_output}"
)

add_executable(aitrain_v2_ocr_acceptance_tests
    tst_v2_ocr_acceptance.cpp
)
target_link_libraries(aitrain_v2_ocr_acceptance_tests
    PRIVATE
        aitrain_v2_application
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql
        Qt${QT_VERSION_MAJOR}::Test
)
set(v2_ocr_acceptance_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_ocr_acceptance_tests_ctest.txt")
add_test(NAME aitrain_v2_ocr_acceptance_tests
    COMMAND aitrain_v2_ocr_acceptance_tests -o "${v2_ocr_acceptance_test_output},txt"
)
set_tests_properties(aitrain_v2_ocr_acceptance_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_ocr_acceptance_test_output}"
)

add_executable(aitrain_v2_ocr_report_packager_tests
    tst_v2_ocr_report_packager.cpp
)
target_link_libraries(aitrain_v2_ocr_report_packager_tests
    PRIVATE
        aitrain_v2_application
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Sql
        Qt${QT_VERSION_MAJOR}::Test
)
set(v2_ocr_report_packager_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_ocr_report_packager_tests_ctest.txt")
add_test(NAME aitrain_v2_ocr_report_packager_tests
    COMMAND aitrain_v2_ocr_report_packager_tests -o "${v2_ocr_report_packager_test_output},txt"
)
set_tests_properties(aitrain_v2_ocr_report_packager_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_ocr_report_packager_test_output}"
)

add_executable(aitrain_v2_annotation_session_tests
    tst_v2_annotation_session.cpp
)
target_link_libraries(aitrain_v2_annotation_session_tests
    PRIVATE
        aitrain_v2_application
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Sql
        Qt${QT_VERSION_MAJOR}::Test
)
set(v2_annotation_session_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_v2_annotation_session_tests_ctest.txt")
add_test(NAME aitrain_v2_annotation_session_tests
    COMMAND aitrain_v2_annotation_session_tests -o "${v2_annotation_session_test_output},txt"
)
set_tests_properties(aitrain_v2_annotation_session_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${v2_annotation_session_test_output}"
)

add_executable(aitrain_delivery_acceptance_ui_tests
    ../src/app/src/AppStyle.h
    ../src/app/src/AppStyle.cpp
    ../src/app/src/DiagnosticBundlePresenterV2.h
    ../src/app/src/DiagnosticBundlePresenterV2.cpp
    ../src/app/src/EnvironmentCheckPresenterV2.h
    ../src/app/src/EnvironmentCheckPresenterV2.cpp
    ../src/app/src/ModelRegistryPresenterV2.h
    ../src/app/src/ModelRegistryPresenterV2.cpp
    ../src/app/src/DatasetConversionUiModel.h
    ../src/app/src/DatasetConversionUiModel.cpp
    ../src/app/src/EvaluationReportView.h
    ../src/app/src/EvaluationReportView.cpp
    ../src/app/src/InfoPanel.h
    ../src/app/src/InfoPanel.cpp
    ../src/app/src/LanguageSupport.h
    ../src/app/src/LanguageSupport.cpp
    ../src/app/src/MainWindow.h
    ../src/app/src/MainWindow.cpp
    ../src/app/src/MainWindowActions.cpp
    ../src/app/src/MainWindowDatasetActions.cpp
    ../src/app/src/MainWindowDeliveryActions.cpp
    ../src/app/src/MainWindowDatasetPage.cpp
    ../src/app/src/MainWindowDeliveryPages.cpp
    ../src/app/src/MainWindowModelActions.cpp
    ../src/app/src/MainWindowModelPages.cpp
    ../src/app/src/MainWindowPages.cpp
    ../src/app/src/MainWindowPreviews.cpp
    ../src/app/src/MainWindowProjectPages.cpp
    ../src/app/src/MainWindowRecords.cpp
    ../src/app/src/MainWindowSummaries.cpp
    ../src/app/src/MainWindowSupport.h
    ../src/app/src/MainWindowSupport.cpp
    ../src/app/src/MainWindowState.h
    ../src/app/src/MainWindowSystemPages.cpp
    ../src/app/src/MainWindowTaskPage.cpp
    ../src/app/src/MainWindowTrainingPage.cpp
    ../src/app/src/MainWindowWorkerMessages.cpp
    ../src/app/src/MetricsWidget.h
    ../src/app/src/MetricsWidget.cpp
    ../src/app/src/RegistrationDialog.h
    ../src/app/src/RegistrationDialog.cpp
    ../src/app/src/ProjectSummaryPresenterV2.h
    ../src/app/src/ProjectSummaryPresenterV2.cpp
    ../src/app/src/Sidebar.h
    ../src/app/src/Sidebar.cpp
    ../src/app/src/StatusPill.h
    ../src/app/src/StatusPill.cpp
    ../src/app/src/TaskArtifactPanel.h
    ../src/app/src/TaskArtifactPanel.cpp
    ../src/app/src/TaskArtifactPresenterV2.h
    ../src/app/src/TaskArtifactPresenterV2.cpp
    ../src/app/src/WorkerClient.h
    ../src/app/src/WorkerClient.cpp
    ../src/app/src/WorkspaceRouter.h
    ../src/app/src/WorkspaceRouter.cpp
    tst_delivery_acceptance_ui.cpp
)

target_include_directories(aitrain_delivery_acceptance_ui_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/app/src
        ${CMAKE_BINARY_DIR}/src/app/generated
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(aitrain_delivery_acceptance_ui_tests
    PRIVATE
        aitrain_core
        aitrain_v2_application
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Widgets
        Qt${QT_VERSION_MAJOR}::Network
        Qt${QT_VERSION_MAJOR}::Test
        Qt${QT_VERSION_MAJOR}::Sql
)

if(AITRAIN_WITH_TENSORRT_SDK)
    target_include_directories(aitrain_delivery_acceptance_ui_tests
        PRIVATE
            ${AITRAIN_TENSORRT_INCLUDE_DIR}
            ${AITRAIN_CUDA_INCLUDE_DIR}
            ${AITRAIN_CUDA_CRT_INCLUDE_DIR}
    )
endif()
if(AITRAIN_WITH_NCNN)
    target_include_directories(aitrain_delivery_acceptance_ui_tests
        PRIVATE
            ${AITRAIN_NCNN_INCLUDE_DIR}
    )
endif()

set(delivery_acceptance_ui_test_output "${CMAKE_BINARY_DIR}/tests/aitrain_delivery_acceptance_ui_tests_ctest.txt")
add_test(NAME aitrain_delivery_acceptance_ui_tests
    COMMAND aitrain_delivery_acceptance_ui_tests -platform offscreen -o "${delivery_acceptance_ui_test_output},txt"
)
set_tests_properties(aitrain_delivery_acceptance_ui_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ATTACHED_FILES_ON_FAIL "${delivery_acceptance_ui_test_output}"
    ENVIRONMENT "${AITRAIN_QT_TEST_ENVIRONMENT}"
)
if(WIN32 AND DEFINED ENV{WINDIR})
    set(delivery_acceptance_ui_test_environment
        ${AITRAIN_QT_TEST_ENVIRONMENT}
        "QT_QPA_FONTDIR=$ENV{WINDIR}/Fonts"
    )
    set_tests_properties(aitrain_delivery_acceptance_ui_tests PROPERTIES
        ENVIRONMENT "${delivery_acceptance_ui_test_environment}"
    )
endif()

aitrain_copy_onnxruntime_runtime(aitrain_delivery_acceptance_ui_tests)
aitrain_copy_ncnn_runtime(aitrain_delivery_acceptance_ui_tests)
aitrain_copy_tensorrt_runtime(aitrain_delivery_acceptance_ui_tests)
aitrain_copy_qt_platform_runtime(aitrain_delivery_acceptance_ui_tests)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME aitrain_yolo_telemetry_python_tests
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_yolo_telemetry.py"
    )
    set_tests_properties(aitrain_yolo_telemetry_python_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
    add_test(NAME aitrain_smp_semantic_python_tests
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_smp_semantic_segmentation.py"
    )
    set_tests_properties(aitrain_smp_semantic_python_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
    add_test(NAME aitrain_anomalib_adapter_python_tests
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_anomalib_adapter.py"
    )
    set_tests_properties(aitrain_anomalib_adapter_python_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
    add_test(NAME aitrain_adapter_sdk_python_tests
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_adapter_sdk.py"
    )
    set_tests_properties(aitrain_adapter_sdk_python_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
    add_test(NAME aitrain_paddleocr_v2_python_tests
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_paddleocr_v2.py"
    )
    set_tests_properties(aitrain_paddleocr_v2_python_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
endif()
