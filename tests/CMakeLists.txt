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
endfunction()

add_aitrain_qt_test(aitrain_platform_tests tst_platform.cpp)
add_aitrain_qt_test(aitrain_repository_workflow_tests tst_repository_workflow.cpp)
add_aitrain_qt_test(aitrain_dataset_tests tst_datasets.cpp)
add_aitrain_qt_test(aitrain_dataset_conversion_tests tst_dataset_conversion.cpp)
add_aitrain_qt_test(aitrain_dataset_conversion_ui_tests tst_dataset_conversion_ui.cpp)
add_aitrain_qt_test(aitrain_detection_inference_tests tst_detection_inference.cpp)
add_aitrain_qt_test(aitrain_ocr_segmentation_worker_tests tst_ocr_segmentation_worker.cpp)
