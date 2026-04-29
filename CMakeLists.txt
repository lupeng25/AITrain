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
option(AITRAIN_ENABLE_ONNXRUNTIME "Enable ONNX Runtime inference when an SDK is available" ON)
set(AITRAIN_ONNXRUNTIME_ROOT "" CACHE PATH "Optional ONNX Runtime SDK root containing include and lib directories")

if(MSVC)
    add_compile_options(/utf-8)
endif()

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Gui Widgets Network Sql Test)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Widgets Network Sql Test)

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

add_subdirectory(src/core)
add_subdirectory(src/app)
add_subdirectory(src/worker)
add_subdirectory(src/plugins/dataset_interop)
add_subdirectory(src/plugins/yolo_native)
add_subdirectory(src/plugins/ocr_rec)

if(AITRAIN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
