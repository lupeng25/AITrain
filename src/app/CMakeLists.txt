add_executable(AITrainStudio WIN32
    src/main.cpp
    src/AppStyle.h
    src/AppStyle.cpp
    src/InfoPanel.h
    src/InfoPanel.cpp
    src/MainWindow.h
    src/MainWindow.cpp
    src/MetricsWidget.h
    src/MetricsWidget.cpp
    src/Sidebar.h
    src/Sidebar.cpp
    src/StatusPill.h
    src/StatusPill.cpp
    src/WorkerClient.h
    src/WorkerClient.cpp
)

target_link_libraries(AITrainStudio
    PRIVATE
        aitrain_core
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Widgets
        Qt${QT_VERSION_MAJOR}::Network
        Qt${QT_VERSION_MAJOR}::Sql
)

set_target_properties(AITrainStudio PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

aitrain_copy_onnxruntime_runtime(AITrainStudio)
