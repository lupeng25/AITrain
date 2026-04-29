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

if(MSVC)
    add_compile_options(/utf-8)
endif()

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Gui Widgets Network Sql Test)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Widgets Network Sql Test)

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
