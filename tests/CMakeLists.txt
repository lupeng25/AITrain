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
    )

    aitrain_copy_onnxruntime_runtime(${target})
    aitrain_copy_ncnn_runtime(${target})
    aitrain_copy_tensorrt_runtime(${target})
    aitrain_copy_qt_platform_runtime(${target})
endfunction()

add_aitrain_qt_test(aitrain_platform_tests tst_platform.cpp)
add_aitrain_qt_test(aitrain_repository_workflow_tests tst_repository_workflow.cpp)
add_aitrain_qt_test(aitrain_dataset_tests tst_datasets.cpp)
add_aitrain_qt_test(aitrain_dataset_conversion_tests tst_dataset_conversion.cpp)
add_aitrain_qt_test(aitrain_dataset_conversion_ui_tests tst_dataset_conversion_ui.cpp)
add_aitrain_qt_test(aitrain_detection_inference_tests tst_detection_inference.cpp)
add_aitrain_qt_test(aitrain_ocr_segmentation_worker_tests tst_ocr_segmentation_worker.cpp)

add_executable(aitrain_delivery_acceptance_ui_tests
    ../src/app/src/AppStyle.h
    ../src/app/src/AppStyle.cpp
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
    ../src/app/src/MainWindowQueueActions.cpp
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
    ../src/app/src/PluginMarketplaceWidget.h
    ../src/app/src/PluginMarketplaceWidget.cpp
    ../src/app/src/RegistrationDialog.h
    ../src/app/src/RegistrationDialog.cpp
    ../src/app/src/Sidebar.h
    ../src/app/src/Sidebar.cpp
    ../src/app/src/StatusPill.h
    ../src/app/src/StatusPill.cpp
    ../src/app/src/TaskArtifactPanel.h
    ../src/app/src/TaskArtifactPanel.cpp
    ../src/app/src/WorkerClient.h
    ../src/app/src/WorkerClient.cpp
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
)

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
endif()
