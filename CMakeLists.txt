cmake_minimum_required(VERSION 3.21)

project(AITrainStudio
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

option(AITRAIN_BUILD_TESTS "Build AITrain Studio tests" ON)
option(AITRAIN_BUILD_LICENSE_GENERATOR "Build the AITrain graphical license generator" ON)
option(AITRAIN_INSTALL_LICENSE_GENERATOR "Install the AITrain license generator with the package" OFF)
option(AITRAIN_ENABLE_ONNXRUNTIME "Enable ONNX Runtime inference when an SDK is available" ON)
option(AITRAIN_INSTALL_QT_RUNTIME "Run windeployqt during install on Windows when available" ON)
set(AITRAIN_LICENSE_PUBLIC_KEY "" CACHE STRING "Base64 encoded ECDSA P-256 public key for offline license verification")
set(AITRAIN_ONNXRUNTIME_ROOT "" CACHE PATH "Optional ONNX Runtime SDK root containing include and lib directories")
set(AITRAIN_CUDA_ROOT "" CACHE PATH "Optional CUDA Toolkit root used for packaging runtime DLLs")
set(AITRAIN_CUDNN_ROOT "" CACHE PATH "Optional cuDNN root used for packaging runtime DLLs")
set(AITRAIN_TENSORRT_ROOT "" CACHE PATH "Optional TensorRT SDK root used for packaging runtime DLLs")
set(AITRAIN_WINDEPLOYQT_EXECUTABLE "" CACHE FILEPATH "Optional path to windeployqt for Windows install packaging")

if(MSVC)
    add_compile_options(/utf-8)
endif()

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Gui Widgets Network Sql Test)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Widgets Network Sql Test)

if(WIN32 AND AITRAIN_INSTALL_QT_RUNTIME AND NOT AITRAIN_WINDEPLOYQT_EXECUTABLE)
    if(TARGET Qt${QT_VERSION_MAJOR}::qmake)
        get_target_property(AITRAIN_QMAKE_EXECUTABLE Qt${QT_VERSION_MAJOR}::qmake IMPORTED_LOCATION)
        get_filename_component(AITRAIN_QT_BIN_DIR "${AITRAIN_QMAKE_EXECUTABLE}" DIRECTORY)
    endif()
    find_program(AITRAIN_FOUND_WINDEPLOYQT_EXECUTABLE
        NAMES windeployqt
        HINTS ${AITRAIN_QT_BIN_DIR}
    )
    if(AITRAIN_FOUND_WINDEPLOYQT_EXECUTABLE)
        set(AITRAIN_WINDEPLOYQT_EXECUTABLE "${AITRAIN_FOUND_WINDEPLOYQT_EXECUTABLE}" CACHE FILEPATH "Optional path to windeployqt for Windows install packaging" FORCE)
    endif()
endif()

find_file(AITRAIN_CUDA_RUNTIME_DLL
    NAMES cudart64_13.dll cudart64_12.dll cudart64_120.dll cudart64_110.dll
    HINTS
        ${CMAKE_SOURCE_DIR}/.deps/cuda-redist/cuda_cudart-windows-x86_64-13.0.96/cuda_cudart-windows-x86_64-13.0.96-archive/bin/x64
        ${AITRAIN_CUDA_ROOT}/bin
        $ENV{CUDA_PATH}/bin
)
find_path(AITRAIN_CUDA_INCLUDE_DIR
    NAMES cuda_runtime_api.h
    HINTS
        ${CMAKE_SOURCE_DIR}/.deps/cuda-redist/cuda_cudart-windows-x86_64-13.0.96/cuda_cudart-windows-x86_64-13.0.96-archive/include
        ${AITRAIN_CUDA_ROOT}/include
        $ENV{CUDA_PATH}/include
)
find_path(AITRAIN_CUDA_CRT_INCLUDE_DIR
    NAMES crt/host_defines.h
    HINTS
        ${CMAKE_SOURCE_DIR}/.deps/cuda-redist/cuda_crt-windows-x86_64-13.0.88/cuda_crt-windows-x86_64-13.0.88-archive/include
        ${AITRAIN_CUDA_ROOT}/include
        $ENV{CUDA_PATH}/include
)
find_file(AITRAIN_CUDNN_DLL
    NAMES cudnn64_9.dll cudnn64_8.dll
    HINTS
        ${CMAKE_SOURCE_DIR}/.deps/cudnn-redist/cudnn-windows-x86_64-9.21.1.3_cuda13/cudnn-windows-x86_64-9.21.1.3_cuda13-archive/bin/x64
        ${AITRAIN_CUDNN_ROOT}/bin
        ${AITRAIN_CUDNN_ROOT}/lib
        ${AITRAIN_CUDA_ROOT}/bin
        $ENV{CUDNN_ROOT}/bin
        $ENV{CUDNN_ROOT}/lib
        $ENV{CUDA_PATH}/bin
)
file(GLOB AITRAIN_CUDNN_RUNTIME_DLLS CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/.deps/cudnn-redist/cudnn-windows-x86_64-9.21.1.3_cuda13/cudnn-windows-x86_64-9.21.1.3_cuda13-archive/bin/x64/cudnn*.dll"
    "${AITRAIN_CUDNN_ROOT}/bin/cudnn*.dll"
    "${AITRAIN_CUDNN_ROOT}/lib/cudnn*.dll"
)
find_path(AITRAIN_TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS
        ${CMAKE_SOURCE_DIR}/.deps/tensorrt-oss/include
        ${AITRAIN_TENSORRT_ROOT}/include
        $ENV{TENSORRT_ROOT}/include
        $ENV{TRT_ROOT}/include
)
find_file(AITRAIN_TENSORRT_INFER_DLL
    NAMES nvinfer.dll nvinfer_10.dll nvinfer_8.dll
    HINTS
        ${CMAKE_SOURCE_DIR}/.deps/tensorrt-pypi/extracted/tensorrt_cu13_libs-10.16.1.11-py3-none-win_amd64/tensorrt_libs
        ${AITRAIN_TENSORRT_ROOT}/bin
        ${AITRAIN_TENSORRT_ROOT}/lib
        ${AITRAIN_TENSORRT_ROOT}/lib/x64
        $ENV{TENSORRT_ROOT}/bin
        $ENV{TENSORRT_ROOT}/lib
        $ENV{TENSORRT_ROOT}/lib/x64
        $ENV{TRT_ROOT}/bin
        $ENV{TRT_ROOT}/lib
        $ENV{TRT_ROOT}/lib/x64
)
find_file(AITRAIN_TENSORRT_PLUGIN_DLL
    NAMES nvinfer_plugin.dll nvinfer_plugin_10.dll nvinfer_plugin_8.dll
    HINTS
        ${CMAKE_SOURCE_DIR}/.deps/tensorrt-pypi/extracted/tensorrt_cu13_libs-10.16.1.11-py3-none-win_amd64/tensorrt_libs
        ${AITRAIN_TENSORRT_ROOT}/bin
        ${AITRAIN_TENSORRT_ROOT}/lib
        ${AITRAIN_TENSORRT_ROOT}/lib/x64
        $ENV{TENSORRT_ROOT}/bin
        $ENV{TENSORRT_ROOT}/lib
        $ENV{TENSORRT_ROOT}/lib/x64
        $ENV{TRT_ROOT}/bin
        $ENV{TRT_ROOT}/lib
        $ENV{TRT_ROOT}/lib/x64
)
find_file(AITRAIN_TENSORRT_ONNX_PARSER_DLL
    NAMES nvonnxparser.dll nvonnxparser_10.dll nvonnxparser_8.dll
    HINTS
        ${CMAKE_SOURCE_DIR}/.deps/tensorrt-pypi/extracted/tensorrt_cu13_libs-10.16.1.11-py3-none-win_amd64/tensorrt_libs
        ${AITRAIN_TENSORRT_ROOT}/bin
        ${AITRAIN_TENSORRT_ROOT}/lib
        ${AITRAIN_TENSORRT_ROOT}/lib/x64
        $ENV{TENSORRT_ROOT}/bin
        $ENV{TENSORRT_ROOT}/lib
        $ENV{TENSORRT_ROOT}/lib/x64
        $ENV{TRT_ROOT}/bin
        $ENV{TRT_ROOT}/lib
        $ENV{TRT_ROOT}/lib/x64
)
file(GLOB AITRAIN_TENSORRT_BUILDER_RESOURCE_DLLS CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/.deps/tensorrt-pypi/extracted/tensorrt_cu13_libs-10.16.1.11-py3-none-win_amd64/tensorrt_libs/nvinfer_builder_resource_*.dll"
    "${AITRAIN_TENSORRT_ROOT}/bin/nvinfer_builder_resource_*.dll"
    "${AITRAIN_TENSORRT_ROOT}/lib/nvinfer_builder_resource_*.dll"
    "${AITRAIN_TENSORRT_ROOT}/lib/x64/nvinfer_builder_resource_*.dll"
)
if(AITRAIN_TENSORRT_INCLUDE_DIR AND AITRAIN_CUDA_INCLUDE_DIR AND AITRAIN_CUDA_CRT_INCLUDE_DIR)
    set(AITRAIN_WITH_TENSORRT_SDK ON)
    message(STATUS "TensorRT SDK found: ${AITRAIN_TENSORRT_INCLUDE_DIR}")
else()
    set(AITRAIN_WITH_TENSORRT_SDK OFF)
    message(STATUS "TensorRT SDK headers not found; TensorRT export/inference remains an unsupported scaffold")
endif()

if(AITRAIN_ENABLE_ONNXRUNTIME)
    find_path(AITRAIN_ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS
            ${CMAKE_SOURCE_DIR}/.deps/onnxruntime-win-x64-1.24.3/include
            ${CMAKE_SOURCE_DIR}/.deps/onnxruntime/include
            ${AITRAIN_ONNXRUNTIME_ROOT}/include
            $ENV{ONNXRUNTIME_ROOT}/include
            $ENV{ORT_ROOT}/include
    )
    find_library(AITRAIN_ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        HINTS
            ${CMAKE_SOURCE_DIR}/.deps/onnxruntime-win-x64-1.24.3/lib
            ${CMAKE_SOURCE_DIR}/.deps/onnxruntime/lib
            ${AITRAIN_ONNXRUNTIME_ROOT}/lib
            ${AITRAIN_ONNXRUNTIME_ROOT}/lib/x64
            $ENV{ONNXRUNTIME_ROOT}/lib
            $ENV{ONNXRUNTIME_ROOT}/lib/x64
            $ENV{ORT_ROOT}/lib
            $ENV{ORT_ROOT}/lib/x64
    )
    find_file(AITRAIN_ONNXRUNTIME_DLL
        NAMES onnxruntime.dll
        HINTS
            ${CMAKE_SOURCE_DIR}/.deps/onnxruntime-win-x64-1.24.3/lib
            ${CMAKE_SOURCE_DIR}/.deps/onnxruntime/lib
            ${AITRAIN_ONNXRUNTIME_ROOT}/lib
            ${AITRAIN_ONNXRUNTIME_ROOT}/lib/x64
            $ENV{ONNXRUNTIME_ROOT}/lib
            $ENV{ONNXRUNTIME_ROOT}/lib/x64
            $ENV{ORT_ROOT}/lib
            $ENV{ORT_ROOT}/lib/x64
    )
    find_file(AITRAIN_ONNXRUNTIME_PROVIDERS_DLL
        NAMES onnxruntime_providers_shared.dll
        HINTS
            ${CMAKE_SOURCE_DIR}/.deps/onnxruntime-win-x64-1.24.3/lib
            ${CMAKE_SOURCE_DIR}/.deps/onnxruntime/lib
            ${AITRAIN_ONNXRUNTIME_ROOT}/lib
            ${AITRAIN_ONNXRUNTIME_ROOT}/lib/x64
            $ENV{ONNXRUNTIME_ROOT}/lib
            $ENV{ONNXRUNTIME_ROOT}/lib/x64
            $ENV{ORT_ROOT}/lib
            $ENV{ORT_ROOT}/lib/x64
    )
    if(AITRAIN_ONNXRUNTIME_INCLUDE_DIR AND AITRAIN_ONNXRUNTIME_LIBRARY AND AITRAIN_ONNXRUNTIME_DLL)
        set(AITRAIN_WITH_ONNXRUNTIME ON)
        message(STATUS "ONNX Runtime SDK found: ${AITRAIN_ONNXRUNTIME_INCLUDE_DIR}")
    else()
        set(AITRAIN_WITH_ONNXRUNTIME OFF)
        message(STATUS "ONNX Runtime SDK not found; ONNX Runtime inference will report unavailable at runtime")
    endif()
endif()

function(aitrain_copy_onnxruntime_runtime target_name)
    if(AITRAIN_WITH_ONNXRUNTIME)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${AITRAIN_ONNXRUNTIME_DLL}
                $<TARGET_FILE_DIR:${target_name}>
            VERBATIM
        )
        if(AITRAIN_ONNXRUNTIME_PROVIDERS_DLL)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${AITRAIN_ONNXRUNTIME_PROVIDERS_DLL}
                    $<TARGET_FILE_DIR:${target_name}>
                VERBATIM
            )
        endif()
    endif()
endfunction()

set(AITRAIN_PACKAGED_ACCELERATOR_DLLS)
foreach(runtime_dll
        ${AITRAIN_CUDA_RUNTIME_DLL}
        ${AITRAIN_CUDNN_DLL}
        ${AITRAIN_CUDNN_RUNTIME_DLLS}
        ${AITRAIN_TENSORRT_INFER_DLL}
        ${AITRAIN_TENSORRT_PLUGIN_DLL}
        ${AITRAIN_TENSORRT_ONNX_PARSER_DLL}
        ${AITRAIN_TENSORRT_BUILDER_RESOURCE_DLLS})
    if(runtime_dll)
        list(APPEND AITRAIN_PACKAGED_ACCELERATOR_DLLS ${runtime_dll})
    endif()
endforeach()
if(AITRAIN_PACKAGED_ACCELERATOR_DLLS)
    list(REMOVE_DUPLICATES AITRAIN_PACKAGED_ACCELERATOR_DLLS)
endif()

function(aitrain_copy_tensorrt_runtime target_name)
    if(AITRAIN_PACKAGED_ACCELERATOR_DLLS)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                $<TARGET_FILE_DIR:${target_name}>/runtimes/tensorrt
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${AITRAIN_PACKAGED_ACCELERATOR_DLLS}
                $<TARGET_FILE_DIR:${target_name}>/runtimes/tensorrt
            VERBATIM
        )
    endif()
endfunction()

add_subdirectory(src/core)
if(AITRAIN_BUILD_LICENSE_GENERATOR)
    add_subdirectory(src/license_generator)
endif()
add_subdirectory(src/app)
add_subdirectory(src/worker)
add_subdirectory(src/plugins/dataset_interop)
add_subdirectory(src/plugins/yolo_native)
add_subdirectory(src/plugins/ocr_rec)

add_custom_target(aitrain_python_trainers_copy ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/python_trainers
        ${CMAKE_BINARY_DIR}/bin/python_trainers
    VERBATIM
)
add_dependencies(AITrainStudio aitrain_python_trainers_copy)
add_dependencies(aitrain_worker aitrain_python_trainers_copy)

install(DIRECTORY DESTINATION plugins/models)
install(DIRECTORY DESTINATION runtimes/onnxruntime)
install(DIRECTORY DESTINATION runtimes/tensorrt)
install(DIRECTORY DESTINATION docs)
install(DIRECTORY docs/harness DESTINATION docs)
install(DIRECTORY docs/acceptance-templates DESTINATION docs)
install(DIRECTORY examples DESTINATION .)
install(FILES
    docs/training-backends.md
    docs/hardware-compatibility.md
    docs/acceptance-runbook.md
    docs/yolo-model-support-matrix.md
    docs/local-rc-closeout.md
    docs/external-acceptance-handoff.md
    docs/release-freeze-handoff.md
    docs/product-roadmap-local-training-platform.md
    DESTINATION docs
)
install(DIRECTORY python_trainers DESTINATION .
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
)
install(FILES
    installer/AITrainStudio.iss
    DESTINATION installer
)
install(FILES
    tools/acceptance-smoke.ps1
    tools/build-inno-installer.ps1
    tools/phase45-yolo-model-matrix-smoke.ps1
    tools/local-rc-closeout.ps1
    tools/release-freeze-handoff.ps1
    tools/materialize-ultralytics-dataset.py
    tools/phase16-ocr-official-smoke.ps1
    tools/phase31-paddleocr-full-official-smoke.ps1
    tools/phase47-paddleocr-det-onnx-smoke.ps1
    tools/production-ocr-acceptance.ps1
    DESTINATION tools
)

if(AITRAIN_WITH_ONNXRUNTIME)
    install(FILES ${AITRAIN_ONNXRUNTIME_DLL} DESTINATION .)
    install(FILES ${AITRAIN_ONNXRUNTIME_DLL} DESTINATION runtimes/onnxruntime)
    if(AITRAIN_ONNXRUNTIME_PROVIDERS_DLL)
        install(FILES ${AITRAIN_ONNXRUNTIME_PROVIDERS_DLL} DESTINATION .)
        install(FILES ${AITRAIN_ONNXRUNTIME_PROVIDERS_DLL} DESTINATION runtimes/onnxruntime)
    endif()
endif()

if(AITRAIN_PACKAGED_ACCELERATOR_DLLS)
    install(FILES ${AITRAIN_PACKAGED_ACCELERATOR_DLLS} DESTINATION runtimes/tensorrt)
endif()

if(WIN32 AND AITRAIN_INSTALL_QT_RUNTIME AND AITRAIN_WINDEPLOYQT_EXECUTABLE)
    install(CODE "
        get_filename_component(AITRAIN_INSTALL_PREFIX_ABSOLUTE \"\${CMAKE_INSTALL_PREFIX}\" ABSOLUTE)
        execute_process(
            COMMAND \"${AITRAIN_WINDEPLOYQT_EXECUTABLE}\" --no-translations \"\${AITRAIN_INSTALL_PREFIX_ABSOLUTE}/AITrainStudio.exe\"
            WORKING_DIRECTORY \"\${AITRAIN_INSTALL_PREFIX_ABSOLUTE}\"
            RESULT_VARIABLE AITRAIN_WINDEPLOYQT_RESULT
        )
        if(NOT AITRAIN_WINDEPLOYQT_RESULT EQUAL 0)
            message(WARNING \"windeployqt failed with exit code \${AITRAIN_WINDEPLOYQT_RESULT}\")
        endif()
    ")
endif()

if(AITRAIN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

set(CPACK_PACKAGE_NAME "AITrainStudio")
set(CPACK_PACKAGE_VENDOR "AITrain")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${PROJECT_VERSION}-win64")
set(CPACK_GENERATOR "ZIP")
include(CPack)
