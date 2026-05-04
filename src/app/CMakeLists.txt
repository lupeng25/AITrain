add_executable(AITrainStudio WIN32
    src/main.cpp
    src/AiTrainAppConfig.h.in
    src/AppStyle.h
    src/AppStyle.cpp
    src/EvaluationReportView.h
    src/EvaluationReportView.cpp
    src/InfoPanel.h
    src/InfoPanel.cpp
    src/LanguageSupport.h
    src/LanguageSupport.cpp
    src/MainWindow.h
    src/MainWindow.cpp
    src/MetricsWidget.h
    src/MetricsWidget.cpp
    src/RegistrationDialog.h
    src/RegistrationDialog.cpp
    src/Sidebar.h
    src/Sidebar.cpp
    src/StatusPill.h
    src/StatusPill.cpp
    src/WorkerClient.h
    src/WorkerClient.cpp
    resources/app.qrc
    resources/app.rc
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/AiTrainAppConfig.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/AiTrainAppConfig.h
    @ONLY
)

target_include_directories(AITrainStudio
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/generated
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

if(TARGET Qt${QT_VERSION_MAJOR}::qmake)
    get_target_property(AITRAIN_APP_QMAKE_EXECUTABLE Qt${QT_VERSION_MAJOR}::qmake IMPORTED_LOCATION)
    get_filename_component(AITRAIN_APP_QT_BIN_DIR "${AITRAIN_APP_QMAKE_EXECUTABLE}" DIRECTORY)
endif()

find_program(AITRAIN_LRELEASE_EXECUTABLE
    NAMES lrelease lrelease-qt5
    HINTS ${AITRAIN_APP_QT_BIN_DIR}
)

set(AITRAIN_TRANSLATION_TS
    ${CMAKE_CURRENT_SOURCE_DIR}/translations/aitrain_zh_CN.ts
    ${CMAKE_CURRENT_SOURCE_DIR}/translations/aitrain_en_US.ts
)
set(AITRAIN_TRANSLATION_QM)
if(AITRAIN_LRELEASE_EXECUTABLE)
    foreach(ts_file IN LISTS AITRAIN_TRANSLATION_TS)
        get_filename_component(ts_name "${ts_file}" NAME_WE)
        set(qm_file "${CMAKE_CURRENT_BINARY_DIR}/translations/${ts_name}.qm")
        add_custom_command(
            OUTPUT "${qm_file}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/translations"
            COMMAND "${AITRAIN_LRELEASE_EXECUTABLE}" "${ts_file}" -qm "${qm_file}"
            DEPENDS "${ts_file}"
            VERBATIM
        )
        list(APPEND AITRAIN_TRANSLATION_QM "${qm_file}")
    endforeach()
    add_custom_target(aitrain_app_translations ALL DEPENDS ${AITRAIN_TRANSLATION_QM})
    add_dependencies(AITrainStudio aitrain_app_translations)
    add_custom_command(TARGET AITrainStudio POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:AITrainStudio>/translations"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${AITRAIN_TRANSLATION_QM} "$<TARGET_FILE_DIR:AITrainStudio>/translations"
        VERBATIM
    )
    install(FILES ${AITRAIN_TRANSLATION_QM} DESTINATION translations)
else()
    message(WARNING "lrelease was not found; AITrain translation .qm files will not be generated")
endif()

aitrain_copy_onnxruntime_runtime(AITrainStudio)
aitrain_copy_tensorrt_runtime(AITrainStudio)

install(TARGETS AITrainStudio
    RUNTIME DESTINATION .
)
