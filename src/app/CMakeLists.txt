add_executable(AITrainStudio WIN32
    src/main.cpp
    src/AiTrainAppConfig.h.in
    src/AppStyle.h
    src/AppStyle.cpp
    src/ApplicationEventRouter.h
    src/ApplicationEventRouter.cpp
    src/ApplicationSettingsService.h
    src/ApplicationSettingsService.cpp
    src/DatasetConversionUiModel.h
    src/DatasetConversionUiModel.cpp
    src/DatasetCatalogPresenter.h
    src/DatasetCatalogPresenter.cpp
    src/DashboardPage.h
    src/DashboardPage.cpp
    src/DashboardPageController.h
    src/DashboardPageController.cpp
    src/DatasetPage.h
    src/DatasetPage.cpp
    src/DatasetPageController.h
    src/DatasetPageController.cpp
    src/DeliveryEvidencePresenter.h
    src/DeliveryEvidencePresenter.cpp
    src/DeliveryEvidencePage.h
    src/DeliveryEvidencePage.cpp
    src/DeliveryEvidencePageController.h
    src/DeliveryEvidencePageController.cpp
    src/EvaluationReportView.h
    src/EvaluationReportView.cpp
    src/DiagnosticBundlePresenter.h
    src/DiagnosticBundlePresenter.cpp
    src/EnvironmentCheckPresenter.h
    src/EnvironmentCheckPresenter.cpp
    src/EnvironmentPage.h
    src/EnvironmentPage.cpp
    src/EnvironmentPageController.h
    src/EnvironmentPageController.cpp
    src/InfoPanel.h
    src/InfoPanel.cpp
    src/LanguageSupport.h
    src/LanguageSupport.cpp
    src/LicenseTokenStore.h
    src/LicenseTokenStore.cpp
    src/MainWindow.h
    src/MainWindow.cpp
    src/MainWindowActions.cpp
    src/MainWindowDatasetActions.cpp
    src/MainWindowDatasetPage.cpp
    src/MainWindowRuntimePage.cpp
    src/MainWindowDeliveryEvidencePage.cpp
    src/MainWindowModelPages.cpp
    src/MainWindowPages.cpp
    src/MainWindowPreviews.cpp
    src/MainWindowProjectPages.cpp
    src/MainWindowRecords.cpp
    src/MainWindowSummaries.cpp
    src/MainWindowSupport.h
    src/MainWindowSupport.cpp
    src/MainWindowState.h
    src/MainWindowSystemPages.cpp
    src/MainWindowTaskPage.cpp
    src/MainWindowTrainingPage.cpp
    src/MainWindowWorkerMessages.cpp
    src/MetricsWidget.h
    src/MetricsWidget.cpp
    src/ModelRegistryPresenter.h
    src/ModelRegistryPresenter.cpp
    src/ModelRegistryPage.h
    src/ModelRegistryPage.cpp
    src/ModelRegistryPageController.h
    src/ModelRegistryPageController.cpp
    src/RegistrationDialog.h
    src/RegistrationDialog.cpp
    src/RuntimeDeliveryPage.h
    src/RuntimeDeliveryPage.cpp
    src/RuntimeDeliveryPageController.h
    src/RuntimeDeliveryPageController.cpp
    src/ProjectSummaryPresenter.h
    src/ProjectSummaryPresenter.cpp
    src/ProjectSessionController.h
    src/ProjectSessionController.cpp
    src/ProjectPage.h
    src/ProjectPage.cpp
    src/ProjectPageController.h
    src/ProjectPageController.cpp
    src/SettingsPage.h
    src/SettingsPage.cpp
    src/SettingsPageController.h
    src/SettingsPageController.cpp
    src/Sidebar.h
    src/Sidebar.cpp
    src/StatusPill.h
    src/StatusPill.cpp
    src/TaskArtifactPanel.h
    src/TaskArtifactPanel.cpp
    src/TaskArtifactPage.h
    src/TaskArtifactPage.cpp
    src/TaskArtifactPageController.h
    src/TaskArtifactPageController.cpp
    src/TaskArtifactTableModels.h
    src/TaskArtifactTableModels.cpp
    src/TaskArtifactPresenter.h
    src/TaskArtifactPresenter.cpp
    src/TaskRuntimeController.h
    src/TaskRuntimeController.cpp
    src/TrainingPage.h
    src/TrainingPage.cpp
    src/TrainingPageController.h
    src/TrainingPageController.cpp
    src/WorkspaceReadModelCoordinator.h
    src/WorkspaceReadModelCoordinator.cpp
    src/WorkerClient.h
    src/WorkerClient.cpp
    src/WorkspaceRouter.h
    src/WorkspaceRouter.cpp
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
        aitrain_vision_data
        aitrain_annotation_integration
        aitrain_dataset_conversion
        aitrain_model_export
        aitrain_vision_runtime
        aitrain_capability_registry
        aitrain_license
        aitrain_worker_protocol
        aitrain_application
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
    install(FILES ${AITRAIN_TRANSLATION_QM}
        DESTINATION translations
        COMPONENT Runtime)
else()
    message(WARNING "lrelease was not found; AITrain translation .qm files will not be generated")
endif()

aitrain_copy_onnxruntime_runtime(AITrainStudio)
aitrain_copy_ncnn_runtime(AITrainStudio)
aitrain_copy_tensorrt_runtime(AITrainStudio)
aitrain_copy_qt_platform_runtime(AITrainStudio)

install(TARGETS AITrainStudio
    RUNTIME DESTINATION .
    COMPONENT Runtime
)
