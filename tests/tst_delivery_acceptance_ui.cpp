#include "MainWindow.h"
#include "AppStyle.h"
#include "ApplicationSettingsService.h"
#include "WorkbenchTranslation.h"
#include "SettingsPage.h"
#include "SettingsPageController.h"
#include "StatusPill.h"
#include <QProcess>
#include <QRegularExpression>
#include <QMessageBox>
#include <QAbstractItemModel>
#include <QJsonDocument>
#include <QListWidget>
#include <QScrollBar>
#include "ApplicationEventRouter.h"
#include "DashboardPage.h"
#include "DashboardPageController.h"
#include "MainWindowSupport.h"
#include "ProjectSessionController.h"
#include "Sidebar.h"
#include "DatasetPage.h"
#include "DatasetPageController.h"
#include "TrainingPage.h"
#include "TrainingPageController.h"
#include "RuntimeDeliveryPage.h"
#include "DeliveryEvidencePage.h"
#include "TaskArtifactTableModels.h"
#include "TaskRuntimeController.h"
#include "ModelRegistryPage.h"
#include "aitrain/product/ProductCapabilityContract.h"
#include "TaskArtifactPanel.h"
#include "TaskArtifactPage.h"
#include "TaskArtifactPageController.h"
#include "TaskArtifactPresenter.h"
#include "WorkerClient.h"
#include "WorkspaceRouter.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/workflow/ProjectWorkspace.h"
#include "aitrain/workflow/TrainingWorkflowProfile.h"

#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QFrame>
#include <QImage>
#include <QScrollArea>
#include <QLabel>
#include <QLineEdit>
#include <QMetaObject>
#include <QPushButton>
#include <QPlainTextEdit>
#include <QSettings>
#include <QSignalSpy>
#include <QStackedWidget>
#include <QStringList>
#include <QTabWidget>
#include <QTableView>
#include <QTableWidget>
#include <QTest>
#include <QThread>
#include <QTemporaryDir>
#include <QTimer>
#include <QTranslator>
#include <QUuid>

#include "LanguageSupport.h"

class EnvironmentDeliveryEvidenceUiTests : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void mainNavigationUsesThreeDailyEntries();
    void businessPagesUseSingleModeViews();
    void evidenceHasIndependentModelToolEntry();
    void environmentPageUsesRuntimeReportRows();
    void clickingSidebarEnvironmentSwitchesPageWithoutDeliveryEntry();
    void capabilityPanelEnglishTranslationIsComplete();
    void workerStartFailureIsReportedAsynchronously();
    void runtimeDeliveryPagesExposeOneSixStepProductEntry();
    void annotationSessionUiUsesArtifactBoundary();
    void dataQualityUiUsesRegisteredIdentityBoundary();
    void datasetConversionUiUsesExplicitImportBoundary();
    void datasetSnapshotImportUiUsesExplicitImportBoundary();
    void datasetSplitUiUsesRegisteredIdentityBoundary();
    void ocrAcceptanceUiUsesControlledImportAndArtifactOnlyAcceptance();
    void taskPageExposesReadOnlyObjects();
    void projectAndDashboardExposeSummaryPresenter();
    void dashboardUsesIndependentRecentTenQuery();
    void uiPathBoundariesStayAtExplicitImportAndIdentityEdges();
    void repeatedNavigationDoesNotAccumulatePages();
    void reviewSamplePathsRejectExternalAndTraversalValues();
    void largeArtifactsRequireExplicitSelectionAndSkipSynchronousPreview();
    void datasetFormatProbeRunsOffUiThreadAndReturnsOnContextThread();
    void projectOpenProbeRunsRecoveryOffUiThreadAndReturnsPreparedToken();
    void projectSessionOperationsDoNotInferCreateFromMissingPath();
    void projectOpenPreparedTokenRejectsMutationWithoutClosingCurrentWorkspace();
    void eventRouterUsesBoundedRollingWindowsAndEvictsTerminalTasks();
    void trainingDraftKeepsBackendAndModelAcrossNavigation();
    void advancedCancelRestoresDraftAndAllOfficialBackendsRemain();
    void unloadedArtifactInventoryIsNotReportedAsZero();
    void reopenedDatasetShowsNameSamplesAndQualityResult();
    void workbenchViewsFitStandardWindows();
    void draftPersistenceAcrossProcessesAndInvalidBindings();
    void englishAndDarkWorkbenchAreComplete();

private:
    QString previousLanguage_;
    bool hadPreviousLanguage_ = false;
    MainWindow* window_ = nullptr;
    QTranslator translator_;
};

void EnvironmentDeliveryEvidenceUiTests::dashboardUsesIndependentRecentTenQuery()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.createProject(
        directory.filePath(QStringLiteral("dashboard-project")), &error),
        qPrintable(error));
    for (int index = 0; index < 12; ++index) {
        aitrain::TaskSnapshot task;
        QVERIFY2(workspace.startTask(aitrain::TaskId::create(),
            QStringLiteral("dashboard.%1").arg(index),
            QStringLiteral("diagnostics"), &task, &error), qPrintable(error));
    }

    aitrain::ProjectQueryService queryService(&workspace);
    DashboardPageController controller(&queryService);
    DashboardWorkspacePage page;
    controller.attachPage(&page);
    controller.setContext(true, QStringLiteral("dashboard-project"),
        8, QStringLiteral("通过"), QStringLiteral("GPU 已检测"));
    controller.refresh();

    auto* recent = page.findChild<QTableWidget*>(
        QStringLiteral("DashboardRecentTasks"));
    auto* total = page.findChild<QLabel*>(
        QStringLiteral("DashboardTaskSummary"));
    QVERIFY(recent != nullptr);
    QVERIFY(total != nullptr);
    QCOMPARE(recent->rowCount(), 10);
    QCOMPARE(total->text(), QStringLiteral("12"));
}

void EnvironmentDeliveryEvidenceUiTests::reviewSamplePathsRejectExternalAndTraversalValues()
{
    QJsonObject unsafe;
    unsafe.insert(QStringLiteral("sampleRelativePath"), QStringLiteral("C:/private/image.jpg"));
    unsafe.insert(QStringLiteral("sourceRelativePath"), QStringLiteral("../../labels/secret.txt"));
    const aitrain_app::ReviewSamplePathView rejected = aitrain_app::reviewSamplePathView(unsafe);
    QVERIFY(rejected.imageRelativePath.isEmpty());
    QVERIFY(rejected.labelRelativePath.isEmpty());

    QJsonObject safe;
    safe.insert(QStringLiteral("sampleRelativePath"), QStringLiteral("images\\train\\a.jpg"));
    safe.insert(QStringLiteral("sourceRelativePath"), QStringLiteral("labels/train/a.txt"));
    const aitrain_app::ReviewSamplePathView accepted = aitrain_app::reviewSamplePathView(safe);
    QCOMPARE(accepted.imageRelativePath, QStringLiteral("images/train/a.jpg"));
    QCOMPARE(accepted.labelRelativePath, QStringLiteral("labels/train/a.txt"));
}

void EnvironmentDeliveryEvidenceUiTests::largeArtifactsRequireExplicitSelectionAndSkipSynchronousPreview()
{
    TaskArtifactPanel panel;
    TaskArtifactDetails details;
    ArtifactFileItem artifact;
    artifact.artifactId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    artifact.kind = QStringLiteral("model");
    artifact.relativePath = QStringLiteral("model.onnx");
    artifact.byteCount = 32LL * 1024LL * 1024LL;
    details.artifacts.append(artifact);
    details.artifactFiles.append(artifact);
    panel.setDetails(details);

    auto* table = panel.findChild<QTableView*>(QStringLiteral("TaskArtifactFileTable"));
    auto* tabs = panel.findChild<QStackedWidget*>(QStringLiteral("TaskDetailViews"));
    auto* preview = panel.findChild<QPlainTextEdit*>(QStringLiteral("ArtifactPreviewText"));
    QVERIFY(table != nullptr);
    QVERIFY(tabs != nullptr);
    QVERIFY(preview != nullptr);
    QVERIFY(!table->currentIndex().isValid());

    tabs->setCurrentIndex(4);
    table->selectRow(0);
    QCoreApplication::processEvents();
    // 预览入口不再执行同步文件读取；没有绑定查询服务时只显示即时错误，
    // 不应再出现“跳过同步预览”这一旧边界文案。
    QVERIFY(!preview->toPlainText().contains(QStringLiteral("已跳过同步预览")));
    QVERIFY(preview->toPlainText().contains(QStringLiteral("查询服务不可用")));
}

void EnvironmentDeliveryEvidenceUiTests::datasetFormatProbeRunsOffUiThreadAndReturnsOnContextThread()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString root = directory.path();
    QVERIFY(QDir().mkpath(QDir(root).filePath(QStringLiteral("images/train"))));
    QVERIFY(QDir().mkpath(QDir(root).filePath(QStringLiteral("labels/train"))));

    QFile yaml(QDir(root).filePath(QStringLiteral("data.yaml")));
    QVERIFY(yaml.open(QIODevice::WriteOnly | QIODevice::Text));
    QVERIFY(yaml.write("path: .\nnames: [item]\n") > 0);
    yaml.close();
    QFile image(QDir(root).filePath(QStringLiteral("images/train/sample.jpg")));
    QVERIFY(image.open(QIODevice::WriteOnly));
    QVERIFY(image.write("not-a-real-image") > 0);
    image.close();
    QFile label(QDir(root).filePath(QStringLiteral("labels/train/sample.txt")));
    QVERIFY(label.open(QIODevice::WriteOnly | QIODevice::Text));
    QVERIFY(label.write("0 0.5 0.5 0.25 0.25\n") > 0);
    label.close();

    QObject context;
    QEventLoop loop;
    QTimer timeout;
    timeout.setSingleShot(true);
    timeout.setInterval(5000);
    QString detected;
    QThread* callbackThread = nullptr;
    connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
    aitrain_app::detectDatasetFormatAsync(&context, root,
        [&](const QString& format) {
            detected = format;
            callbackThread = QThread::currentThread();
            loop.quit();
        });
    timeout.start();
    loop.exec();

    QCOMPARE(detected, QStringLiteral("yolo_detection"));
    QCOMPARE(callbackThread, QCoreApplication::instance()->thread());
}

void EnvironmentDeliveryEvidenceUiTests::projectOpenProbeRunsRecoveryOffUiThreadAndReturnsPreparedToken()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());

    QObject context;
    QEventLoop loop;
    QTimer timeout;
    timeout.setSingleShot(true);
    timeout.setInterval(15000);
    aitrain_app::ProjectOpenProbeResult result;
    QThread* callbackThread = nullptr;
    connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
    aitrain_app::probeProjectOpenAsync(&context,
        aitrain_app::ProjectSessionOperation::Create, directory.path(),
        [&](const aitrain_app::ProjectOpenProbeResult& value) {
            result = value;
            callbackThread = QThread::currentThread();
            loop.quit();
        });
    timeout.start();
    loop.exec();

    QVERIFY2(result.succeeded, qPrintable(result.error));
    QVERIFY(result.prepared.isValid());
    QCOMPARE(callbackThread, QCoreApplication::instance()->thread());

    // 仅把值类型 prepared token 带回 GUI，再由 GUI 线程建立自己的 SQL 连接。
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.openPrepared(result.prepared, &error), qPrintable(error));
    QVERIFY(workspace.isOpen());
    workspace.close();
}

void EnvironmentDeliveryEvidenceUiTests::projectSessionOperationsDoNotInferCreateFromMissingPath()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString missingRoot = directory.filePath(QStringLiteral("missing-project"));
    QVERIFY(!QFileInfo::exists(missingRoot));

    const auto runProbe = [&](aitrain_app::ProjectSessionOperation operation) {
        aitrain_app::ProjectOpenProbeResult result;
        QEventLoop loop;
        QTimer timeout;
        timeout.setSingleShot(true);
        timeout.setInterval(15000);
        connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
        aitrain_app::probeProjectOpenAsync(&loop, operation, missingRoot,
            [&](const aitrain_app::ProjectOpenProbeResult& value) {
                result = value;
                loop.quit();
            });
        timeout.start();
        loop.exec();
        return result;
    };

    const aitrain_app::ProjectOpenProbeResult openResult =
        runProbe(aitrain_app::ProjectSessionOperation::Open);
    QVERIFY(!openResult.succeeded);
    QVERIFY(!QFileInfo::exists(missingRoot));

    const aitrain_app::ProjectOpenProbeResult createResult =
        runProbe(aitrain_app::ProjectSessionOperation::Create);
    QVERIFY2(createResult.succeeded, qPrintable(createResult.error));
    QVERIFY(createResult.prepared.isValid());
    QVERIFY(QFileInfo::exists(
        QDir(missingRoot).filePath(QStringLiteral(".aitrain/project.sqlite"))));

    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.openPrepared(createResult.prepared, &error), qPrintable(error));
    workspace.close();
}

void EnvironmentDeliveryEvidenceUiTests::projectOpenPreparedTokenRejectsMutationWithoutClosingCurrentWorkspace()
{
    QTemporaryDir first;
    QTemporaryDir second;
    QVERIFY(first.isValid());
    QVERIFY(second.isValid());

    aitrain::ProjectWorkspace current;
    QString error;
    QVERIFY2(current.createProject(first.path(), &error), qPrintable(error));

    aitrain::ProjectWorkspace secondInitializer;
    QVERIFY2(secondInitializer.createProject(second.path(), &error), qPrintable(error));
    const QString secondDatabase = QDir(secondInitializer.workspacePath())
        .filePath(QStringLiteral("project.sqlite"));
    secondInitializer.close();

    aitrain::PreparedProjectSession prepared;
    QVERIFY2(aitrain::ProjectWorkspace::prepareOpen(second.path(), &prepared, &error),
        qPrintable(error));
    QVERIFY(prepared.isValid());

    // 激活只复核 ProjectId + openGeneration，不再读取 DB/WAL/staging 全量
    // 指纹。模拟票据生成后 generation 变化，候选必须被拒绝且当前 Session
    // 保持原样。
    aitrain::ProjectStore concurrentStore;
    QVERIFY2(concurrentStore.open(secondDatabase, &error), qPrintable(error));
    aitrain::ProjectMetaSnapshot advanced;
    QVERIFY2(concurrentStore.advanceOpenGeneration(&advanced, &error), qPrintable(error));
    concurrentStore.close();

    QVERIFY(!current.openPrepared(prepared, &error));
    QCOMPARE(current.workspacePath(),
        QDir(first.path()).filePath(QStringLiteral(".aitrain")));
    QVERIFY(current.isOpen());
    current.close();
}

void EnvironmentDeliveryEvidenceUiTests::eventRouterUsesBoundedRollingWindowsAndEvictsTerminalTasks()
{
    namespace wp = aitrain::worker_protocol;
    WorkerClient worker;
    ApplicationEventRouter router(&worker);
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    wp::TaskEvent event;
    event.taskId = taskId;
    event.kind = wp::TaskEventKind::Metric;
    event.details.insert(QStringLiteral("name"), QStringLiteral("loss"));
    for (int i = 0; i < 1100; ++i) {
        event.details.insert(QStringLiteral("value"), i);
        QVERIFY(QMetaObject::invokeMethod(&router, "onTaskEvent", Qt::DirectConnection,
            Q_ARG(aitrain::worker_protocol::TaskEvent, event)));
    }
    TaskViewState state = router.viewState(taskId.toString());
    QCOMPARE(state.metricSequence, qint64(1100));
    QCOMPARE(state.metrics.size(), 1024);
    QCOMPARE(state.metrics.constLast().value, 1099.0);

    event.kind = wp::TaskEventKind::Artifact;
    for (int i = 0; i < 300; ++i) {
        event.details = QJsonObject{
            {QStringLiteral("artifactId"), QString::number(i)},
            {QStringLiteral("kind"), QStringLiteral("checkpoint")},
            {QStringLiteral("relativePath"), QStringLiteral("checkpoints/%1.pt").arg(i)}};
        QVERIFY(QMetaObject::invokeMethod(&router, "onTaskEvent", Qt::DirectConnection,
            Q_ARG(aitrain::worker_protocol::TaskEvent, event)));
    }
    state = router.viewState(taskId.toString());
    QCOMPARE(state.artifactSequence, qint64(300));
    QCOMPARE(state.artifacts.size(), 256);
    QCOMPARE(state.artifacts.constLast().artifactId, QStringLiteral("299"));

    event.kind = wp::TaskEventKind::Succeeded;
    event.details = QJsonObject{{QStringLiteral("message"), QStringLiteral("done")}};
    QVERIFY(QMetaObject::invokeMethod(&router, "onTaskEvent", Qt::DirectConnection,
        Q_ARG(aitrain::worker_protocol::TaskEvent, event)));
    QVERIFY(router.viewState(taskId.toString()).taskId.isEmpty());
}

void EnvironmentDeliveryEvidenceUiTests::initTestCase()
{
    const QString isolatedSettings = qEnvironmentVariable("AITRAIN_DRAFT_TEST_ROOT");
    if (!isolatedSettings.isEmpty()) {
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, isolatedSettings);
    }
    QApplication::setStyle(QStringLiteral("Fusion"));
    AppStyle::apply(*qApp, QStringLiteral("light"));
    QCoreApplication::setOrganizationName(QStringLiteral("AITrainTests"));
    QCoreApplication::setApplicationName(QStringLiteral("DeliveryAcceptanceUiTests"));
    QSettings settings;
    previousLanguage_ = settings.value(aitrain_app::languageSettingsKey()).toString();
    hadPreviousLanguage_ = settings.contains(aitrain_app::languageSettingsKey());
    aitrain_app::storeLanguageCode(QStringLiteral("zh_CN"));

    QApplication::setEffectEnabled(Qt::UI_AnimateCombo, false);
    QApplication::setEffectEnabled(Qt::UI_AnimateMenu, false);
    QApplication::setEffectEnabled(Qt::UI_FadeMenu, false);
    QApplication::setEffectEnabled(Qt::UI_AnimateTooltip, false);
    QApplication::setEffectEnabled(Qt::UI_FadeTooltip, false);
    window_ = new MainWindow(QStringLiteral("test-license"), QStringLiteral("2099-12-31"));
    const QString capturePath = qEnvironmentVariable("AITRAIN_CAPTURE_UI_PATH");
    if (!capturePath.isEmpty()) {
        QDir().mkpath(QFileInfo(capturePath).absolutePath());
        window_->resize(1280, 800);
        bool pageOk = false;
        const int capturePage = qEnvironmentVariableIntValue("AITRAIN_CAPTURE_UI_PAGE", &pageOk);
        if (pageOk && capturePage >= MainWindow::DashboardPage && capturePage < MainWindow::PageCount) {
            const QStringList titles = {
                QStringLiteral("总览"),
                QStringLiteral("项目"),
                QStringLiteral("数据集"),
                QStringLiteral("训练实验"),
                QStringLiteral("任务与产物"),
                QStringLiteral("模型库"),
                QStringLiteral("部署验证"),
                QStringLiteral("环境"),
                QStringLiteral("系统设置")
            };
            QVERIFY(QMetaObject::invokeMethod(window_, "showPage", Qt::DirectConnection,
                Q_ARG(int, capturePage), Q_ARG(QString, titles.at(capturePage))));
        }
        bool tabOk = false;
        const int captureTab = qEnvironmentVariableIntValue("AITRAIN_CAPTURE_UI_TAB", &tabOk);
        if (tabOk) {
            QTabWidget* tabs = nullptr;
            switch (capturePage) {
            case MainWindow::DatasetPage: tabs = window_->findChild<QTabWidget*>(QStringLiteral("DatasetTabs")); break;
            case MainWindow::TrainingPage: tabs = window_->findChild<QTabWidget*>(QStringLiteral("TrainingDetailTabs")); break;
            case MainWindow::TaskQueuePage: tabs = window_->findChild<QTabWidget*>(QStringLiteral("TaskDetailTabs")); break;
            case MainWindow::ModelRegistryPage: tabs = window_->findChild<QTabWidget*>(QStringLiteral("ModelWorkspaceTabs")); break;
            case MainWindow::DeploymentPage: tabs = window_->findChild<QTabWidget*>(QStringLiteral("DeploymentTabs")); break;
            case MainWindow::EnvironmentPage: tabs = window_->findChild<QTabWidget*>(QStringLiteral("EnvironmentTabs")); break;
            case MainWindow::SystemSettingsPage: tabs = window_->findChild<QTabWidget*>(QStringLiteral("SystemSettingsTabs")); break;
            default: break;
            }
            if (tabs && captureTab >= 0 && captureTab < tabs->count()) {
                tabs->setCurrentIndex(captureTab);
            }
        }
        bool expandAdvancedOk = false;
        const int expandAdvanced = qEnvironmentVariableIntValue("AITRAIN_CAPTURE_UI_EXPAND_ADVANCED", &expandAdvancedOk);
        if (expandAdvancedOk && expandAdvanced != 0 && pageOk && capturePage == MainWindow::DeploymentPage
            && window_->findChild<QTabWidget*>(QStringLiteral("DeploymentTabs"))) {
            auto* deploymentTabs = window_->findChild<QTabWidget*>(QStringLiteral("DeploymentTabs"));
            if (auto* toggle = deploymentTabs->currentWidget()->findChild<QPushButton*>(QStringLiteral("AdvancedToggle"))) {
                toggle->setChecked(true);
            }
        }
        window_->setAttribute(Qt::WA_ShowWithoutActivating, true);
        window_->show();
        QTest::qWait(150);
        QVERIFY2(window_->grab().save(capturePath), qPrintable(QStringLiteral("无法保存 UI 截图：%1").arg(capturePath)));
        window_->hide();
    }
}

void EnvironmentDeliveryEvidenceUiTests::cleanupTestCase()
{
    qApp->removeTranslator(&translator_);
    delete window_;
    window_ = nullptr;
    QSettings settings;
    if (hadPreviousLanguage_) {
        settings.setValue(aitrain_app::languageSettingsKey(), previousLanguage_);
    } else {
        settings.remove(aitrain_app::languageSettingsKey());
    }
}

void EnvironmentDeliveryEvidenceUiTests::mainNavigationUsesThreeDailyEntries()
{
    auto* sidebar = window_->findChild<Sidebar*>(QStringLiteral("Sidebar"));
    QVERIFY(sidebar);
    QStringList labels;
    for (auto* button : sidebar->findChildren<QPushButton*>()) if (button->objectName() == QStringLiteral("SidebarButton")) labels << button->property("fullText").toString();
    QCOMPARE(labels, QStringList({QStringLiteral("数据集"), QStringLiteral("训练"), QStringLiteral("模型")}));
    QVERIFY(window_->findChild<QPushButton*>(QStringLiteral("GlobalTaskButton")));
    QVERIFY(window_->findChild<QWidget*>(QStringLiteral("InspectorPanel")) == nullptr);
}

void EnvironmentDeliveryEvidenceUiTests::businessPagesUseSingleModeViews()
{
    for (int index : {MainWindow::DatasetPage, MainWindow::TrainingPage, MainWindow::ModelRegistryPage, MainWindow::DeploymentPage, MainWindow::EvidencePage}) {
        QVERIFY(QMetaObject::invokeMethod(window_, "showPage", Qt::DirectConnection, Q_ARG(int, index), Q_ARG(QString, QStringLiteral("工作区"))));
        auto* stack = window_->findChild<QStackedWidget*>(QStringLiteral("WorkspaceStack"));
        auto* page = stack->currentWidget();
        QVERIFY(page->findChildren<QTabWidget*>().isEmpty());
        QVERIFY(page->findChild<QStackedWidget*>(QStringLiteral("WorkspaceModeStack")));
        QVERIFY(qobject_cast<QScrollArea*>(page) == nullptr);
    }
}

void EnvironmentDeliveryEvidenceUiTests::taskPageExposesReadOnlyObjects()
{
    MainWindow& window = *window_;
    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::TaskQueuePage), Q_ARG(QString, QStringLiteral("任务与产物"))));

    QObject* presenter = window.findChild<QObject*>(QStringLiteral("TaskArtifactPresenter"));
    QVERIFY(presenter != nullptr);
    QVERIFY(presenter->property("taskCount").isValid());
    QVERIFY(presenter->property("selectedTaskId").isValid());
    QVERIFY(window.findChild<QTableView*>(QStringLiteral("TaskQueueTable")) != nullptr);
    QVERIFY(window.findChild<QTableView*>(QStringLiteral("TaskArtifactTable")) != nullptr);
    QVERIFY(window.findChild<QTableView*>(QStringLiteral("TaskArtifactFileTable")) != nullptr);
    QVERIFY(window.findChild<QTableView*>(QStringLiteral("TaskMetricTable")) != nullptr);
    QVERIFY(window.findChild<QTableWidget*>(QStringLiteral("TaskWorkflowTable")) != nullptr);
    auto* cancelButton = window.findChild<QPushButton*>(QStringLiteral("TaskCancelButton"));
    QVERIFY(cancelButton != nullptr);
    QVERIFY(!cancelButton->isEnabled());
    for (QPushButton* button : window.findChildren<QPushButton*>()) {
        QVERIFY2(button->text() != QStringLiteral("复现实验"),
            "任务页不得重新引入未接线的复现实验入口");
    }
    auto* detailViews = window.findChild<QStackedWidget*>(QStringLiteral("TaskDetailViews"));
    QVERIFY(detailViews);
    QCOMPARE(detailViews->count(), 5);
}

void EnvironmentDeliveryEvidenceUiTests::annotationSessionUiUsesArtifactBoundary()
{
    MainWindow& window = *window_;
    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::DatasetPage), Q_ARG(QString, QStringLiteral("数据集"))));
    QVERIFY(window.findChild<QPushButton*>(QStringLiteral("CreateAnnotationSessionButton")) != nullptr);
    QVERIFY(window.findChild<QPushButton*>(QStringLiteral("SyncAnnotationSessionButton")) != nullptr);
    for (QPushButton* button : window.findChildren<QPushButton*>()) {
        QVERIFY2(button->text() != QStringLiteral("打开数据目录"),
            "标注页不得通过任意数据集目录绕过 Annotation Session/Artifact 边界");
    }

    namespace wp = aitrain::worker_protocol;
    const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString repairArtifactId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString sessionArtifactId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QJsonObject createPayload = wp::annotationSessionCreateRequest(taskId,
        QStringLiteral("C:/project"), repairArtifactId, QStringLiteral("C:/work/session"),
        QJsonObject{{QStringLiteral("tool"), QStringLiteral("X-AnyLabeling")}}, QJsonObject());
    QCOMPARE(createPayload.value(QStringLiteral("repairManifestArtifactId")).toString(), repairArtifactId);
    QVERIFY(!createPayload.contains(QStringLiteral("datasetPath")));
    QVERIFY(!createPayload.contains(QStringLiteral("outputPath")));
    QVERIFY(!createPayload.contains(QStringLiteral("reportPath")));
    QVERIFY(!createPayload.contains(QStringLiteral("sessionManifestPath")));

    const QJsonObject syncPayload = wp::annotationSessionSyncRequest(taskId,
        QStringLiteral("C:/project"), sessionArtifactId, QStringLiteral("C:/work/session"), QJsonObject());
    QCOMPARE(syncPayload.value(QStringLiteral("sessionArtifactId")).toString(), sessionArtifactId);
    QVERIFY(!syncPayload.contains(QStringLiteral("datasetPath")));
    QVERIFY(!syncPayload.contains(QStringLiteral("outputPath")));
    QVERIFY(!syncPayload.contains(QStringLiteral("reportPath")));
    QVERIFY(!syncPayload.contains(QStringLiteral("sessionManifestPath")));
}

void EnvironmentDeliveryEvidenceUiTests::dataQualityUiUsesRegisteredIdentityBoundary()
{
    MainWindow& window = *window_;
    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::DatasetPage), Q_ARG(QString, QStringLiteral("数据集"))));
    QVERIFY(window.findChild<QPushButton*>(QStringLiteral("RunDataQualityWorkflowButton")) != nullptr);
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("DataQualityDatasetId")) != nullptr);
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("DataQualityDatasetVersionId")) != nullptr);
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("DataQualitySnapshotId")) != nullptr);
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("DataQualitySnapshotArtifactId")) != nullptr);

    namespace wp = aitrain::worker_protocol;
    const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString datasetId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString versionId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString snapshotId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString artifactId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QJsonObject payload = wp::dataQualityWorkflowRequest(taskId,
        QStringLiteral("C:/project"), datasetId, versionId, snapshotId, artifactId,
        QJsonObject{{QStringLiteral("maxIssues"), 500}});
    QCOMPARE(payload.value(QStringLiteral("datasetId")).toString(), datasetId);
    QCOMPARE(payload.value(QStringLiteral("datasetVersionId")).toString(), versionId);
    QCOMPARE(payload.value(QStringLiteral("snapshotId")).toString(), snapshotId);
    QCOMPARE(payload.value(QStringLiteral("snapshotArtifactId")).toString(), artifactId);
    QVERIFY(!payload.contains(QStringLiteral("datasetPath")));
    QVERIFY(!payload.contains(QStringLiteral("reportPath")));
    QVERIFY(!payload.contains(QStringLiteral("outputPath")));
}

void EnvironmentDeliveryEvidenceUiTests::datasetConversionUiUsesExplicitImportBoundary()
{
    MainWindow& window = *window_;
    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::DatasetPage), Q_ARG(QString, QStringLiteral("数据集"))));
    QVERIFY(window.findChild<QPushButton*>(QStringLiteral("RunDatasetConversionWorkflowButton")) != nullptr);
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("DatasetConversionTargetDatasetId")) != nullptr);
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("DatasetConversionTargetDatasetName")) != nullptr);
    namespace wp = aitrain::worker_protocol;
    const QJsonObject payload = wp::datasetConversionWorkflowRequest(
        QUuid::createUuid().toString(QUuid::WithoutBraces), QStringLiteral("C:/project"),
        QStringLiteral("C:/incoming/annotations.json"), QStringLiteral("coco_json"),
        QStringLiteral("yolo_detection"),
        QUuid::createUuid().toString(QUuid::WithoutBraces), QStringLiteral("导入数据集"),
        QJsonObject{{QStringLiteral("copyImages"), true}});
    QVERIFY(payload.contains(QStringLiteral("sourcePath")));
    QVERIFY(payload.contains(QStringLiteral("targetDatasetId")));
    QVERIFY(payload.contains(QStringLiteral("targetDatasetName")));
    QVERIFY(!payload.contains(QStringLiteral("outputPath")));
    QVERIFY(!payload.contains(QStringLiteral("reportPath")));
    QVERIFY(!payload.contains(QStringLiteral("artifactPath")));
}

void EnvironmentDeliveryEvidenceUiTests::datasetSnapshotImportUiUsesExplicitImportBoundary()
{
    MainWindow& window = *window_;
    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::DatasetPage), Q_ARG(QString, QStringLiteral("数据集"))));
    QVERIFY(window.findChild<QPushButton*>(
        QStringLiteral("RunDatasetSnapshotImportWorkflowButton")) != nullptr);
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("DatasetSnapshotTargetDatasetId")) != nullptr);
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("DatasetSnapshotTargetDatasetName")) != nullptr);
    namespace wp = aitrain::worker_protocol;
    const QJsonObject payload = wp::datasetSnapshotImportWorkflowRequest(
        QUuid::createUuid().toString(QUuid::WithoutBraces), QStringLiteral("C:/project"),
        QStringLiteral("C:/incoming/dataset"), QStringLiteral("yolo_detection"),
        QUuid::createUuid().toString(QUuid::WithoutBraces), QStringLiteral("导入快照"),
        QJsonObject{{QStringLiteral("maxFiles"), 20000}});
    QVERIFY(payload.contains(QStringLiteral("sourcePath")));
    QVERIFY(payload.contains(QStringLiteral("targetDatasetId")));
    QVERIFY(payload.contains(QStringLiteral("targetDatasetName")));
    QVERIFY(!payload.contains(QStringLiteral("datasetPath")));
    QVERIFY(!payload.contains(QStringLiteral("outputPath")));
    QVERIFY(!payload.contains(QStringLiteral("reportPath")));
    QVERIFY(!payload.contains(QStringLiteral("artifactPath")));
}

void EnvironmentDeliveryEvidenceUiTests::datasetSplitUiUsesRegisteredIdentityBoundary()
{
    MainWindow& window = *window_;
    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::DatasetPage), Q_ARG(QString, QStringLiteral("数据集"))));
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("SplitSourceDatasetId")) != nullptr);
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("SplitSourceDatasetVersionId")) != nullptr);
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("SplitSourceSnapshotId")) != nullptr);
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("SplitSourceSnapshotArtifactId")) != nullptr);
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("SplitTargetDatasetId")) != nullptr);
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("SplitTargetDatasetName")) != nullptr);
    namespace wp = aitrain::worker_protocol;
    const QString id = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QJsonObject payload = wp::datasetSplitWorkflowRequest(id,
        QStringLiteral("C:/project"), id, id, id, id, id, QStringLiteral("划分目标"),
        QJsonObject{{QStringLiteral("trainRatio"), 0.8},
            {QStringLiteral("valRatio"), 0.2}, {QStringLiteral("seed"), 42}});
    QVERIFY(payload.contains(QStringLiteral("sourceDatasetId")));
    QVERIFY(payload.contains(QStringLiteral("sourceDatasetVersionId")));
    QVERIFY(payload.contains(QStringLiteral("sourceSnapshotId")));
    QVERIFY(payload.contains(QStringLiteral("sourceSnapshotArtifactId")));
    QVERIFY(payload.contains(QStringLiteral("targetDatasetId")));
    QVERIFY(!payload.contains(QStringLiteral("sourcePath")));
    QVERIFY(!payload.contains(QStringLiteral("datasetPath")));
    QVERIFY(!payload.contains(QStringLiteral("outputPath")));
    QVERIFY(!payload.contains(QStringLiteral("reportPath")));
    QVERIFY(!payload.contains(QStringLiteral("artifactPath")));
}

void EnvironmentDeliveryEvidenceUiTests::ocrAcceptanceUiUsesControlledImportAndArtifactOnlyAcceptance()
{
    MainWindow& window = *window_;
    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::EvidencePage), Q_ARG(QString, QStringLiteral("环境"))));
    QVERIFY(window.findChild<QPushButton*>(QStringLiteral("ImportOcrOfficialReportsButton")) != nullptr);
    QVERIFY(window.findChild<QPushButton*>(QStringLiteral("RunOcrAcceptanceWorkflowButton")) != nullptr);
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("OcrDetRawReportPath")) != nullptr);
    QVERIFY(window.findChild<QLineEdit*>(QStringLiteral("OcrDetReportArtifactId")) != nullptr);

    namespace wp = aitrain::worker_protocol;
    const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString detArtifactId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString recArtifactId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString systemArtifactId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QJsonObject source{{QStringLiteral("reportPath"), QStringLiteral("C:/incoming/report.json")},
        {QStringLiteral("snapshotId"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
        {QStringLiteral("snapshotArtifactId"), QString()}};
    const QJsonObject importPayload = wp::ocrOfficialReportImportRequest(taskId,
        QStringLiteral("C:/project"), source, source, source, QStringLiteral("batch-a"),
        QStringLiteral("line-a"), QStringLiteral("customer_domain"));
    QVERIFY(importPayload.value(QStringLiteral("det")).toObject().contains(QStringLiteral("reportPath")));
    QVERIFY(!importPayload.contains(QStringLiteral("datasetPath")));
    QVERIFY(!importPayload.contains(QStringLiteral("outputPath")));

    const QJsonObject acceptancePayload = wp::ocrAcceptanceWorkflowRequest(taskId,
        QStringLiteral("C:/project"), detArtifactId, recArtifactId, systemArtifactId,
        QJsonObject{{QStringLiteral("minimumSystemAccuracy"), 0.7}});
    QCOMPARE(acceptancePayload.value(QStringLiteral("detReportArtifactId")).toString(), detArtifactId);
    QVERIFY(!acceptancePayload.contains(QStringLiteral("reportPath")));
    QVERIFY(!acceptancePayload.contains(QStringLiteral("datasetPath")));
    QVERIFY(!acceptancePayload.contains(QStringLiteral("outputPath")));
    QVERIFY(!acceptancePayload.contains(QStringLiteral("systemImagesPath")));
}

void EnvironmentDeliveryEvidenceUiTests::projectAndDashboardExposeSummaryPresenter()
{
    MainWindow& window = *window_;
    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::DashboardPage), Q_ARG(QString, QStringLiteral("总览"))));

    QObject* presenter = window.findChild<QObject*>(QStringLiteral("ProjectSummaryPresenter"));
    QVERIFY(presenter != nullptr);
    QVERIFY(window.findChild<QObject*>(QStringLiteral("DiagnosticBundlePresenter")) != nullptr);
    QVERIFY(presenter->property("available").isValid());
    QVERIFY(presenter->property("taskCount").isValid());
    QVERIFY(presenter->property("datasetCount").isValid());
    QVERIFY(presenter->property("modelPackageCount").isValid());
    QVERIFY(window.findChild<QLabel*>(QStringLiteral("DashboardDatasetSummary")) != nullptr);
    QVERIFY(window.findChild<QLabel*>(QStringLiteral("DashboardTaskSummary")) != nullptr);

    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::ProjectPage), Q_ARG(QString, QStringLiteral("项目"))));
    QVERIFY(window.findChild<QLabel*>(QStringLiteral("ProjectDatasetSummary")) != nullptr);
    QVERIFY(window.findChild<QLabel*>(QStringLiteral("ProjectTaskSummary")) != nullptr);
    QVERIFY(window.findChild<QLabel*>(QStringLiteral("ProjectModelPackageSummary")) != nullptr);
    QVERIFY(window.findChild<QPushButton*>(QStringLiteral("ProjectCreateButton")) != nullptr);
    QVERIFY(window.findChild<QPushButton*>(QStringLiteral("ProjectOpenButton")) != nullptr);
    QVERIFY(window.findChild<QPushButton*>(QStringLiteral("ProjectRebuildButton")) != nullptr);
    const auto projectButtons = window.findChildren<QPushButton*>();
    for (const QPushButton* button : projectButtons) {
        QVERIFY(button->text() != QStringLiteral("创建 / 打开项目"));
    }

    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::TaskQueuePage), Q_ARG(QString, QStringLiteral("任务与产物"))));
    QVERIFY(window.findChild<QFrame*>(QStringLiteral("ArtifactActionGrid")) == nullptr);
    const auto buttons = window.findChildren<QPushButton*>();
    for (const QPushButton* button : buttons) {
        QVERIFY(button->text() != QStringLiteral("交付报告"));
    }
}

void EnvironmentDeliveryEvidenceUiTests::evidenceHasIndependentModelToolEntry()
{
    QVERIFY(QMetaObject::invokeMethod(window_, "showPage", Qt::DirectConnection, Q_ARG(int, MainWindow::EnvironmentPage), Q_ARG(QString, QStringLiteral("环境"))));
    auto* stack = window_->findChild<QStackedWidget*>(QStringLiteral("WorkspaceStack"));
    QVERIFY(stack->currentWidget()->findChild<DeliveryEvidenceWorkspacePage*>() == nullptr);
    QVERIFY(stack->currentWidget()->findChild<QTableWidget*>(QStringLiteral("EnvironmentTable")));
    QVERIFY(QMetaObject::invokeMethod(window_, "showPage", Qt::DirectConnection, Q_ARG(int, MainWindow::EvidencePage), Q_ARG(QString, QStringLiteral("验收报告"))));
    auto* evidence = window_->findChild<DeliveryEvidenceWorkspacePage*>();
    QVERIFY(evidence);
    QCOMPARE(evidence->acceptanceTable->rowCount(), 0);
    QVERIFY(!evidence->acceptanceSummaryLabel->text().contains(QStringLiteral("通过")));
}

void EnvironmentDeliveryEvidenceUiTests::clickingSidebarEnvironmentSwitchesPageWithoutDeliveryEntry()
{
    window_->resize(1280, 820); window_->show();
    QVERIFY(QTest::qWaitForWindowExposed(window_));
    auto* sidebar = window_->findChild<Sidebar*>(QStringLiteral("Sidebar")); QVERIFY(sidebar);
    QPushButton* environment = nullptr;
    for (auto* button : sidebar->findChildren<QPushButton*>()) if (button->text().contains(QStringLiteral("环境"))) environment = button;
    QVERIFY(environment); QVERIFY(environment->isVisible()); environment->click();
    QCOMPARE(window_->findChild<QStackedWidget*>(QStringLiteral("WorkspaceStack"))->currentIndex(), int(MainWindow::EnvironmentPage));
    QVERIFY(window_->findChild<QTabWidget*>(QStringLiteral("EnvironmentTabs")) == nullptr);
}

void EnvironmentDeliveryEvidenceUiTests::environmentPageUsesRuntimeReportRows()
{
    MainWindow& window = *window_;
    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::EnvironmentPage), Q_ARG(QString, QStringLiteral("环境"))));
    auto* table = window.findChild<QTableWidget*>(QStringLiteral("EnvironmentTable"));
    QVERIFY(table != nullptr);
    QVERIFY(table->rowCount() > 0);
    for (int row = 0; row < table->rowCount(); ++row) {
        QVERIFY(table->item(row, 0) != nullptr);
        QVERIFY(table->item(row, 0)->text() != QStringLiteral("LibTorch"));
        QVERIFY(table->item(row, 1) != nullptr);
        QCOMPARE(table->item(row, 1)->data(Qt::UserRole).toString(), QStringLiteral("unchecked"));
    }
}

void EnvironmentDeliveryEvidenceUiTests::capabilityPanelEnglishTranslationIsComplete()
{
    const QStringList sources = {
        QStringLiteral("编译期注册的模型、数据集、导出和推理能力。"),
        QStringLiteral("能力来源：编译期内置注册表"),
        QStringLiteral("能力由编译期注册表提供。"),
        QStringLiteral("编译期注册"),
        QStringLiteral("GPU 策略"),
        QStringLiteral("GPU 推荐或必需能力"),
        QStringLiteral("运行策略"),
        QStringLiteral("内置"),
        QStringLiteral("检查 NVIDIA 驱动、CUDA、TensorRT、ONNX Runtime、Qt 运行时模块和 Worker 可用性，并集中查看交付证据、诊断包和客户域 OCR 验收。")
    };
    const QStringList expected = {
        QStringLiteral("Model, dataset, export, and inference capabilities registered at compile time."),
        QStringLiteral("Capability source: compile-time built-in registry"),
        QStringLiteral("Capabilities are provided by the compile-time registry."),
        QStringLiteral("Registered at compile time"),
        QStringLiteral("GPU Policy"),
        QStringLiteral("Capabilities that recommend or require a GPU"),
        QStringLiteral("Runtime Policy"),
        QStringLiteral("Built-in"),
        QStringLiteral("Check NVIDIA driver, CUDA, TensorRT, ONNX Runtime, Qt runtime modules, and Worker availability, and review delivery evidence, diagnostics bundles, and customer-domain OCR acceptance in one place.")
    };

    aitrain_app::storeLanguageCode(QStringLiteral("en_US"));
    QVERIFY(aitrain_app::loadTranslator(*qApp, &translator_, QStringLiteral("en_US")));
    QVERIFY(!translator_.isEmpty());
    QStringList actual;
    for (const QString& source : sources) {
        actual.append(aitrain_app::translateText("MainWindow", source));
    }
    aitrain_app::storeLanguageCode(QStringLiteral("zh_CN"));

    qApp->removeTranslator(&translator_);
    QCOMPARE(actual, expected);
}

void EnvironmentDeliveryEvidenceUiTests::workerStartFailureIsReportedAsynchronously()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString invalidWorker = directory.filePath(QStringLiteral("not-a-worker.txt"));
    QFile file(invalidWorker);
    QVERIFY(file.open(QIODevice::WriteOnly));
    QVERIFY(file.write("not an executable") > 0);
    file.close();

    WorkerClient client;
    QSignalSpy finishedSpy(&client, &WorkerClient::finished);
    bool workerLostReceived = false;
    aitrain::TaskId lostTaskId;
    connect(&client, &WorkerClient::workerLost, this,
        [&](const aitrain::TaskId& taskId) {
        workerLostReceived = true;
        lostTaskId = taskId;
    });
    QString error;
    QElapsedTimer elapsed;
    elapsed.start();
    const aitrain::TaskId requestedTaskId = aitrain::TaskId::create();
    const aitrain::worker_protocol::TaskCommand command{
        aitrain::worker_protocol::EnvironmentCheckCommand{
            {requestedTaskId, directory.path()}}};
    QVERIFY2(client.startTask(invalidWorker, command, &error), qPrintable(error));
    QVERIFY2(elapsed.elapsed() < 500, "Worker 启动请求不应同步等待进程创建结果。");
    QVERIFY2(finishedSpy.wait(5000), "异步启动失败必须通过 finished 信号收口。");
    QCOMPARE(finishedSpy.count(), 1);
    QCOMPARE(qvariant_cast<WorkerClient::WorkerTerminalStatus>(finishedSpy.first().at(0)),
        WorkerClient::WorkerTerminalStatus::Failed);
    QVERIFY(finishedSpy.first().at(1).toString().contains(QStringLiteral("failed to start"), Qt::CaseInsensitive));
    QVERIFY(workerLostReceived);
    QCOMPARE(lostTaskId, requestedTaskId);
}

void EnvironmentDeliveryEvidenceUiTests::runtimeDeliveryPagesExposeOneSixStepProductEntry()
{
    QVERIFY(QMetaObject::invokeMethod(window_, "showPage", Qt::DirectConnection, Q_ARG(int, MainWindow::DeploymentPage), Q_ARG(QString, QStringLiteral("验证与交付"))));
    auto* page = window_->findChild<RuntimeDeliveryWorkspacePage*>(); QVERIFY(page);
    QCOMPARE(page->findChildren<QPushButton*>(QStringLiteral("RuntimeDeliveryStart")).size(), 1);
    QVERIFY(page->findChild<QPushButton*>(QStringLiteral("DeliverySelectSample")));
    QVERIFY(page->findChildren<QLineEdit*>().isEmpty());
    QVERIFY(page->formData(RuntimeDeliveryMode::DeploymentValidation).sampleSnapshotId.isEmpty());
    QVERIFY(page->selectedModelPackageId(RuntimeDeliveryMode::DeploymentValidation).isEmpty());
}

void EnvironmentDeliveryEvidenceUiTests::uiPathBoundariesStayAtExplicitImportAndIdentityEdges()
{
    MainWindow& window = *window_;
    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::DatasetPage), Q_ARG(QString, QStringLiteral("数据集"))));
    auto* datasetPath = window.findChild<QLineEdit*>(QStringLiteral("DatasetPathEdit"));
    QVERIFY(datasetPath != nullptr);
    const QString externalPath = QStringLiteral("C:/external/secret-dataset");
    datasetPath->setText(externalPath);

    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::TrainingPage), Q_ARG(QString, QStringLiteral("训练实验"))));
    auto* trainingNote = window.findChild<QLabel*>(QStringLiteral("TrainingDatasetNote"));
    auto* datasetDetail = window.findChild<QLabel*>(QStringLiteral("DatasetDetailLabel"));
    QVERIFY(trainingNote != nullptr);
    QVERIFY(datasetDetail != nullptr);
    QVERIFY(!trainingNote->text().contains(externalPath));
    QVERIFY(!trainingNote->toolTip().contains(externalPath));
    QVERIFY(!datasetDetail->text().contains(externalPath));
    QVERIFY(!datasetDetail->text().contains(QStringLiteral("路径：")));

    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::DashboardPage), Q_ARG(QString, QStringLiteral("总览"))));
    auto* projectStatus = window.findChild<QLabel*>(QStringLiteral("ProjectWorkspaceStatus"));
    QVERIFY(projectStatus != nullptr);
    QVERIFY(!projectStatus->text().contains(QStringLiteral("C:/")));
    QVERIFY(!projectStatus->toolTip().contains(QStringLiteral("C:/")));
}

void EnvironmentDeliveryEvidenceUiTests::repeatedNavigationDoesNotAccumulatePages()
{
    MainWindow& window = *window_;
    const QList<int> pages = {
        MainWindow::DashboardPage, MainWindow::ProjectPage, MainWindow::DatasetPage,
        MainWindow::TrainingPage, MainWindow::TaskQueuePage, MainWindow::ModelRegistryPage,
        MainWindow::DeploymentPage, MainWindow::EnvironmentPage, MainWindow::SystemSettingsPage
    };
    QStackedWidget* stack = window.findChild<QStackedWidget*>();
    QVERIFY(stack != nullptr);
    const int initialCount = stack->count();
    for (int cycle = 0; cycle < 1000; ++cycle) {
        for (const int page : pages) {
            QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
                Q_ARG(int, page), Q_ARG(QString, QStringLiteral("stress"))));
        }
        if ((cycle % 25) == 0) {
            QCoreApplication::processEvents(QEventLoop::AllEvents, 1);
        }
    }
    QCOMPARE(stack->count(), initialCount);
    QCOMPARE(stack->currentIndex(), pages.last());
}

void EnvironmentDeliveryEvidenceUiTests::trainingDraftKeepsBackendAndModelAcrossNavigation()
{
    QVERIFY(QMetaObject::invokeMethod(window_, "showPage", Qt::DirectConnection, Q_ARG(int, MainWindow::TrainingPage), Q_ARG(QString, QStringLiteral("训练"))));
    auto* controller = window_->findChild<TrainingPageController*>(); auto* page = window_->findChild<TrainingWorkspacePage*>(); QVERIFY(controller); QVERIFY(page);
    TrainingDatasetBinding binding; binding.datasetFormat = QStringLiteral("anomaly_folder"); controller->setDatasetBinding(binding);
    auto* backend = page->findChild<QComboBox*>(QStringLiteral("TrainingBackend")); backend->setCurrentIndex(backend->findData(QStringLiteral("anomalib_efficientad")));
    auto* model = page->findChild<QComboBox*>(QStringLiteral("TrainingModelPreset")); model->setCurrentText(QStringLiteral("draft-model"));
    auto* epoch = page->findChild<QLineEdit*>(QStringLiteral("TrainingEpochs")); epoch->setText(QStringLiteral("37"));
    page->setMode(TrainingWorkspacePage::Configuration);
    for (int index : {MainWindow::DatasetPage, MainWindow::TaskQueuePage, MainWindow::ModelRegistryPage, MainWindow::TrainingPage}) QVERIFY(QMetaObject::invokeMethod(window_, "showPage", Qt::DirectConnection, Q_ARG(int, index), Q_ARG(QString, QStringLiteral("工作区"))));
    QCOMPARE(backend->currentData().toString(), QStringLiteral("anomalib_efficientad")); QCOMPARE(model->currentText(), QStringLiteral("draft-model")); QCOMPARE(epoch->text(), QStringLiteral("37"));
    QCOMPARE(page->views->currentIndex(), int(TrainingWorkspacePage::Configuration));
}

void EnvironmentDeliveryEvidenceUiTests::advancedCancelRestoresDraftAndAllOfficialBackendsRemain()
{
    TaskRuntimeController runtime; TrainingPageController controller(&runtime); TrainingWorkspacePage page; controller.attach(&page);
    auto* backends = page.findChild<QComboBox*>(QStringLiteral("TrainingBackend")); QCOMPARE(backends->count(), 8);
    for (const auto& item : aitrain::ProductCapabilityContract::instance().trainingBackends()) {
        TrainingDatasetBinding binding; binding.datasetFormat = item.datasetFormat; controller.setDatasetBinding(binding);
        const int index = backends->findData(item.id); QVERIFY(index >= 0); backends->setCurrentIndex(index); QCOMPARE(page.formData().taskType, item.taskType); QCOMPARE(page.formData().capabilityId, item.capabilityId);
    }
    TrainingDatasetBinding binding; binding.datasetFormat = QStringLiteral("yolo_detection"); controller.setDatasetBinding(binding);
    auto* lr = page.findChild<QLineEdit*>(QStringLiteral("YoloTrainArg_lr0")); QVERIFY(lr); lr->setText(QStringLiteral("0.012"));
    QVERIFY(QMetaObject::invokeMethod(&page, "advancedRequested", Qt::DirectConnection)); lr->setText(QStringLiteral("0.2"));
    QVERIFY(QMetaObject::invokeMethod(&page, "cancelAdvancedRequested", Qt::DirectConnection)); QCOMPARE(lr->text(), QStringLiteral("0.012"));
    page.resize(1000, 700); page.show();
    QVERIFY(QMetaObject::invokeMethod(&page, "advancedRequested", Qt::DirectConnection)); lr->setText(QStringLiteral("0.3"));
    QTest::keyClick(lr, Qt::Key_Escape);
    QCOMPARE(page.views->currentIndex(), int(TrainingWorkspacePage::Configuration)); QCOMPARE(lr->text(), QStringLiteral("0.012"));
    QTest::keyClick(&page, Qt::Key_Escape); QCOMPARE(page.views->currentIndex(), int(TrainingWorkspacePage::Catalog));
}

void EnvironmentDeliveryEvidenceUiTests::unloadedArtifactInventoryIsNotReportedAsZero()
{
    ArtifactTableModel model; ArtifactFileItem item; item.artifactId = aitrain::ArtifactId::create().toString(); item.kind = QStringLiteral("dataset_quality_report");
    model.setFiles({item}); QCOMPARE(model.rowCount(), 1);
    QCOMPARE(model.data(model.index(0, 2), Qt::DisplayRole).toString(), QStringLiteral("文件清单尚未读取"));
}

void EnvironmentDeliveryEvidenceUiTests::reopenedDatasetShowsNameSamplesAndQualityResult()
{
    QTemporaryDir directory; QVERIFY(directory.isValid());
    const QString source = directory.filePath(QStringLiteral("source"));
    const auto write = [](const QString& path, const QByteArray& text) { QDir().mkpath(QFileInfo(path).absolutePath()); QFile file(path); return file.open(QIODevice::WriteOnly) && file.write(text) == text.size(); };
    QVERIFY(write(QDir(source).filePath(QStringLiteral("data.yaml")), QByteArray("path: .\ntrain: images/train\nval: images/val\nnames: [item]\n")));
    QImage image(64, 64, QImage::Format_RGB32); image.fill(Qt::green);
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val")}) {
        const QString file = QDir(source).filePath(QStringLiteral("images/%1/sample.png").arg(split)); QDir().mkpath(QFileInfo(file).absolutePath()); QVERIFY(image.save(file));
        QVERIFY(write(QDir(source).filePath(QStringLiteral("labels/%1/sample.txt").arg(split)), QByteArray("0 0.5 0.5 0.25 0.25\n")));
    }
    const QString root = directory.filePath(QStringLiteral("project")); aitrain::ProjectWorkspace workspace; QString error;
    QVERIFY2(workspace.createProject(root, &error), qPrintable(error));
    aitrain::TaskSnapshot task; const auto importId = aitrain::TaskId::create();
    QVERIFY2(workspace.startTask(importId, QStringLiteral("dataset.snapshot.import"), QStringLiteral("dataset_snapshot_import"), &task, &error), qPrintable(error));
    aitrain::DatasetSnapshotImportWorkflowRequest request; request.sourcePath = source; request.sourceFormat = QStringLiteral("yolo_detection"); request.targetDatasetId = aitrain::DatasetId::create(); request.targetDatasetName = QStringLiteral("表面缺陷样本");
    aitrain::DatasetSnapshotImportWorkflowResult imported;
    QVERIFY2(workspace.runDatasetSnapshotImportWorkflow(importId, request, &imported, &error), qPrintable(error));
    QCOMPARE(imported.terminalState, aitrain::TaskState::Succeeded);
    workspace.close(); QVERIFY2(workspace.open(root, &error), qPrintable(error));
    aitrain::ProjectQueryService query(&workspace); TaskRuntimeController runtime; DatasetPageController controller(&query, &runtime);
    controller.setProjectContext(true, root); DatasetWorkspacePage page; controller.attach(&page);
    QCOMPARE(page.datasetListTable->rowCount(), 1); QCOMPARE(page.datasetListTable->item(0, 0)->text(), request.targetDatasetName);
    QCOMPARE(controller.state().currentSnapshotId, imported.datasetSnapshot.id.toString());
    QCOMPARE(page.datasetPreviewTable->rowCount(), 2); QCOMPARE(page.datasetListTable->item(0, 3)->text(), QStringLiteral("2"));
    QVERIFY(!controller.state().currentSampleRelativePath.isEmpty());
    page.resize(1000, 600); page.show(); page.showView(DatasetWorkspacePage::Detail);
    controller.previewSample(0);
    QTRY_VERIFY2_WITH_TIMEOUT(page.sampleImageLabel->pixmap() && !page.sampleImageLabel->pixmap()->isNull(),
        qPrintable(QStringLiteral("图像预览：%1；区域 %2 x %3").arg(page.sampleImageLabel->text()).arg(page.sampleImageLabel->width()).arg(page.sampleImageLabel->height())), 10000);
    const auto qualityId = aitrain::TaskId::create();
    QVERIFY2(workspace.startTask(qualityId, QStringLiteral("dataset.quality"), QStringLiteral("data_quality"), &task, &error), qPrintable(error));
    QVERIFY(QMetaObject::invokeMethod(&controller, "taskStarted", Qt::DirectConnection, Q_ARG(QString, qualityId.toString()), Q_ARG(QString, QStringLiteral("data_quality"))));
    aitrain::DataQualityWorkflowRequest quality; quality.snapshotId = imported.datasetSnapshot.id; quality.datasetId = imported.datasetSnapshot.datasetId; quality.datasetVersionId = imported.datasetSnapshot.datasetVersionId; quality.snapshotArtifactId = imported.datasetSnapshot.artifactId;
    aitrain::DataQualityWorkflowResult checked; QVERIFY2(workspace.runDataQualityWorkflow(qualityId, quality, &checked, &error), qPrintable(error));
    TaskViewState state; state.taskId = qualityId.toString(); state.status = QStringLiteral("succeeded"); state.terminal = true; controller.applyTaskViewState(state);
    QTRY_VERIFY_WITH_TIMEOUT(page.validationSummaryLabel->text().contains(QStringLiteral("质量检查完成")), 10000);
    QVERIFY(!controller.state().latestQualityArtifactId.isEmpty()); QVERIFY(!page.validationSummaryLabel->text().contains(QStringLiteral("等待")));
    TaskArtifactPage taskPage; TaskArtifactPageController taskController(&query); taskController.attachPage(&taskPage); taskController.openTask(qualityId.toString());
    auto* artifacts = taskPage.findChild<QTableView*>(QStringLiteral("TaskArtifactTable"));
    QVERIFY(artifacts && artifacts->model()->rowCount() > 0); artifacts->selectRow(0);
    auto* files = taskPage.findChild<QTableView*>(QStringLiteral("TaskArtifactFileTable"));
    QVERIFY(files && files->model()->rowCount() > 0); files->selectRow(0);
    const QString member = taskPage.selectedArtifactMember(); QVERIFY(!member.isEmpty());
    QVERIFY(taskController.refreshSelected()); QCOMPARE(taskPage.selectedArtifactMember(), member);
    QVERIFY(artifacts->model()->index(0, 2).data().toString().startsWith(QStringLiteral("已读取")));
    // 从已登记的训练工作流恢复草稿，不启动 Worker，也不创建 Resume 请求。
    const auto trainingId = aitrain::TaskId::create();
    QVERIFY2(workspace.startTask(trainingId, QStringLiteral("yolo"), QStringLiteral("detection"), &task, &error), qPrintable(error));
    aitrain::TrainingWorkflowProfile profile;
    QVERIFY(aitrain::resolveTrainingWorkflowProfile(QStringLiteral("ultralytics_yolo_detect"), &profile, &error));
    aitrain::TrainingWorkflowRequest trainingRequest;
    trainingRequest.templateId = profile.templateId; trainingRequest.datasetId = imported.datasetSnapshot.datasetId; trainingRequest.datasetVersionId = imported.datasetSnapshot.datasetVersionId;
    trainingRequest.snapshotId = imported.datasetSnapshot.id; trainingRequest.snapshotArtifactId = imported.datasetSnapshot.artifactId;
    trainingRequest.trainingBackend = profile.trainingBackend; trainingRequest.evaluationBackend = profile.evaluationBackend; trainingRequest.exportBackend = profile.exportBackend; trainingRequest.deploymentBackend = profile.deploymentBackend;
    trainingRequest.parameterSummary = {{QStringLiteral("epochs"), 7}, {QStringLiteral("batchSize"), 2}, {QStringLiteral("imageSize"), 320}, {QStringLiteral("modelPreset"), QStringLiteral("yolov8s.pt")},
        {QStringLiteral("ultralyticsTrainArgs"), QJsonObject{{QStringLiteral("seed"), 17}, {QStringLiteral("lr0"), 0.002}}}};
    aitrain::TrainingWorkflowDispatch dispatch;
    QVERIFY2(workspace.beginTrainingWorkflow(trainingId, trainingRequest, &dispatch, &error), qPrintable(error));
    TrainingPageController trainingController(&runtime); trainingController.setQueryService(&query); trainingController.setProjectContext(true, root);
    TrainingWorkspacePage trainingPage; trainingController.attach(&trainingPage);
    QCOMPARE(trainingPage.historyTable->rowCount(), 1);
    QVERIFY(QMetaObject::invokeMethod(&trainingPage, "historyRequested", Qt::DirectConnection, Q_ARG(QString, trainingId.toString())));
    QVERIFY(!trainingPage.modelsButton->isEnabled());
    QVERIFY(QMetaObject::invokeMethod(&trainingPage, "copyConfigurationRequested", Qt::DirectConnection));
    QCOMPARE(trainingPage.views->currentIndex(), int(TrainingWorkspacePage::Configuration));
    QCOMPARE(trainingPage.formData().modelPreset, QStringLiteral("yolov8s.pt")); QCOMPARE(trainingPage.formData().epochs, 7);
    QCOMPARE(trainingPage.formData().batchSize, 2); QCOMPARE(trainingPage.formData().imageSize, 320);
    QCOMPARE(trainingPage.findChild<QLineEdit*>(QStringLiteral("YoloTrainArg_lr0"))->text(), QStringLiteral("0.002"));
    QCOMPARE(trainingPage.findChild<QLineEdit*>(QStringLiteral("YoloTrainArg_seed"))->text(), QStringLiteral("17"));
    QVERIFY(!runtime.isRunning());

    const auto versions = query.datasetSnapshots(imported.datasetSnapshot.datasetId, {1, {}}, &error);
    QVERIFY2(error.isEmpty(), qPrintable(error)); QCOMPARE(versions.items.size(), 1); QCOMPARE(versions.items.first().latestQualityTaskId, qualityId);

    // 通过产品的 Session 入口重新打开磁盘项目，验证延迟建页和顶层导航也能恢复数据。
    workspace.close();
    MainWindow reopened(QStringLiteral("test-license"), QStringLiteral("2099-12-31"));
    reopened.setAttribute(Qt::WA_ShowWithoutActivating, true); reopened.resize(1280, 820); reopened.show();
    auto* session = reopened.findChild<ProjectSessionController*>(); QVERIFY(session);
    QSignalSpy failures(session, &ProjectSessionController::failed);
    QVERIFY2(session->request(aitrain_app::ProjectSessionOperation::Open, QStringLiteral("project"), root, &error), qPrintable(error));
    QTRY_VERIFY_WITH_TIMEOUT(!session->isBusy(), 15000);
    QVERIFY2(session->isOpen(), failures.isEmpty() ? "项目未打开" : qPrintable(failures.first().first().toString()));
    auto* realPage = reopened.findChild<DatasetWorkspacePage*>(); QVERIFY(realPage);
    QCOMPARE(realPage->datasetListTable->rowCount(), 1);
    QCOMPARE(realPage->datasetListTable->item(0, 0)->text(), request.targetDatasetName);
    realPage->showView(DatasetWorkspacePage::Detail);
    QTRY_VERIFY_WITH_TIMEOUT(realPage->sampleImageLabel->pixmap() && !realPage->sampleImageLabel->pixmap()->isNull(), 10000);
    const QString capture = qEnvironmentVariable("AITRAIN_CAPTURE_UI_DIR");
    if (!capture.isEmpty()) {
        QDir().mkpath(capture);
        QVERIFY(reopened.grab().save(QDir(capture).filePath(QStringLiteral("真实项目-样本预览.png"))));
        QVERIFY(QMetaObject::invokeMethod(&reopened, "showPage", Qt::DirectConnection, Q_ARG(int, MainWindow::TrainingPage), Q_ARG(QString, QStringLiteral("训练"))));
        auto* realTraining = reopened.findChild<TrainingWorkspacePage*>(); QVERIFY(realTraining);
        QVERIFY(QMetaObject::invokeMethod(realTraining, "historyRequested", Qt::DirectConnection, Q_ARG(QString, trainingId.toString())));
        QVERIFY(QMetaObject::invokeMethod(realTraining, "copyConfigurationRequested", Qt::DirectConnection));
        QCoreApplication::processEvents();
        QVERIFY(reopened.grab().save(QDir(capture).filePath(QStringLiteral("真实项目-历史配置草稿.png"))));
    }
}

void EnvironmentDeliveryEvidenceUiTests::workbenchViewsFitStandardWindows()
{
    MainWindow window(QStringLiteral("test-license"), QStringLiteral("2099-12-31"));
    window.setAttribute(Qt::WA_ShowWithoutActivating, true); window.show();
    const QString output = qEnvironmentVariable("AITRAIN_CAPTURE_UI_DIR");
    if (!output.isEmpty()) QDir().mkpath(output);
    const QStringList titles = {QStringLiteral("项目概况"), QStringLiteral("项目"), QStringLiteral("数据集"), QStringLiteral("训练"), QStringLiteral("任务记录"), QStringLiteral("模型"), QStringLiteral("验证与交付"), QStringLiteral("环境与诊断"), QStringLiteral("设置"), QStringLiteral("验收报告")};
    const QList<QSize> sizes = qEnvironmentVariableIntValue("AITRAIN_TEST_MINIMUM_WINDOW")
        ? QList<QSize>{QSize(1024, 700)} : QList<QSize>{QSize(1280, 820), QSize(1366, 768), QSize(1024, 700)};
    for (const QSize& size : sizes) {
        window.resize(size);
        for (int index = 0; index < MainWindow::PageCount; ++index) {
            QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection, Q_ARG(int, index), Q_ARG(QString, titles.at(index))));
            auto* stack = window.findChild<QStackedWidget*>(QStringLiteral("WorkspaceStack")); QWidget* page = stack->currentWidget();
            auto* host = dynamic_cast<aitrain_app::WorkspaceViewHost*>(page);
            const int modes = host ? host->views->count() : 1;
            for (int mode = 0; mode < modes; ++mode) {
                if (host) host->setMode(mode);
                QCoreApplication::processEvents();
                QCOMPARE(window.size(), size);
                for (auto* button : page->findChildren<QPushButton*>()) {
                    if (!button->isVisibleTo(page)) continue;
                    const QRect bounds(button->mapTo(&window, QPoint()), button->size());
                    QVERIFY2(window.rect().contains(bounds), qPrintable(QStringLiteral("按钮超出窗口：page %1 / mode %2 / %3 / %4,%5 %6x%7").arg(index).arg(mode).arg(button->text()).arg(bounds.x()).arg(bounds.y()).arg(bounds.width()).arg(bounds.height())));
                }
                if (!output.isEmpty()) QVERIFY(window.grab().save(QDir(output).filePath(QStringLiteral("%1x%2-page%3-mode%4.png").arg(size.width()).arg(size.height()).arg(index).arg(mode))));
            }
        }
    }
}

void EnvironmentDeliveryEvidenceUiTests::draftPersistenceAcrossProcessesAndInvalidBindings()
{
    const QString phase = qEnvironmentVariable("AITRAIN_DRAFT_TEST_PHASE");
    if (phase.isEmpty()) {
        QTemporaryDir directory; QVERIFY(directory.isValid());
        for (const QString& mode : {QStringLiteral("write"), QStringLiteral("read")}) {
            QProcess child; auto environment = QProcessEnvironment::systemEnvironment();
            environment.insert(QStringLiteral("AITRAIN_DRAFT_TEST_ROOT"), directory.path());
            environment.insert(QStringLiteral("AITRAIN_DRAFT_TEST_PHASE"), mode); child.setProcessEnvironment(environment);
            const QString log = directory.filePath(mode + QStringLiteral(".txt"));
            child.start(QCoreApplication::applicationFilePath(), {QStringLiteral("draftPersistenceAcrossProcessesAndInvalidBindings"), QStringLiteral("-platform"), QStringLiteral("offscreen"), QStringLiteral("-o"), log + QStringLiteral(",txt")});
            QTRY_VERIFY_WITH_TIMEOUT(child.state() == QProcess::NotRunning, 60000);
            QFile output(log); output.open(QIODevice::ReadOnly);
            QVERIFY2(child.exitStatus() == QProcess::NormalExit && child.exitCode() == 0, output.readAll().constData());
        }
        return;
    }
    const QString base = qEnvironmentVariable("AITRAIN_DRAFT_TEST_ROOT"), root = QDir(base).filePath(QStringLiteral("project"));
    aitrain::ProjectWorkspace workspace; QString error;
    aitrain::ProjectQueryService query(&workspace); TaskRuntimeController runtime;
    TrainingPageController controller(&runtime); TrainingWorkspacePage page; controller.setQueryService(&query); controller.attach(&page);
    auto* epoch = page.findChild<QLineEdit*>(QStringLiteral("TrainingEpochs"));
    auto* model = page.findChild<QComboBox*>(QStringLiteral("TrainingModelPreset"));
    auto* lr = page.findChild<QLineEdit*>(QStringLiteral("YoloTrainArg_lr0"));
    auto* status = page.findChild<QLabel*>(QStringLiteral("TrainingDraftStatus"));
    auto* save = page.findChild<QPushButton*>(QStringLiteral("TrainingSaveDraft"));
    aitrain_app::ApplicationSettingsService settings;
    if (phase == QStringLiteral("write")) {
        QVERIFY2(workspace.createProject(root, &error), qPrintable(error));
        aitrain::ProjectStore store; QVERIFY(store.open(QDir(root).filePath(QStringLiteral(".aitrain/project.sqlite")), &error));
        store.setArtifactStoreRoot(QDir(root).filePath(QStringLiteral(".aitrain/artifacts")));
        aitrain::TaskSnapshot task; task.id = aitrain::TaskId::create(); task.requestId = aitrain::RequestId::create(); task.capabilityId = QStringLiteral("dataset"); task.taskType = QStringLiteral("dataset_snapshot"); QVERIFY(store.createTask(task, &error));
        const auto artifact = aitrain::ArtifactId::create();
        QVERIFY2(store.recordArtifactWithFiles(artifact, task.id, QStringLiteral("dataset_snapshot"),
            {{QStringLiteral("dataset_snapshot.json"), QString(64, QLatin1Char('a')), 2}, {QStringLiteral("images/a.png"), QString(64, QLatin1Char('b')), 10}}, QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
        aitrain::DatasetSnapshotRecord snapshot; snapshot.id = aitrain::SnapshotId::create(); snapshot.datasetId = aitrain::DatasetId::create(); snapshot.taskId = task.id; snapshot.artifactId = artifact;
        snapshot.rootPath = QDir(root).filePath(QStringLiteral(".aitrain/artifacts/committed/%1").arg(artifact.toString())); snapshot.datasetFormat = QStringLiteral("yolo_detection"); snapshot.driverId = snapshot.datasetFormat; snapshot.driverVersion = QStringLiteral("2.0"); snapshot.rootHash = QString(64, QLatin1Char('b')); snapshot.manifestSha256 = QString(64, QLatin1Char('a')); snapshot.fileCount = 2; snapshot.totalBytes = 12; snapshot.createdAt = QDateTime::currentDateTimeUtc();
        QVERIFY2(store.registerDatasetSnapshot(&snapshot, &error), qPrintable(error));
        controller.setProjectContext(true, root);
        TrainingDatasetBinding binding; binding.datasetId = snapshot.datasetId.toString(); binding.datasetVersionId = snapshot.datasetVersionId.toString(); binding.snapshotId = snapshot.id.toString(); binding.snapshotArtifactId = artifact.toString(); binding.datasetFormat = snapshot.datasetFormat; binding.displayName = QStringLiteral("中文用户数据"); binding.deploymentSampleRelativePath = QStringLiteral("images/a.png"); controller.setDatasetBinding(binding);
        epoch->setText(QStringLiteral("37")); model->setCurrentText(QStringLiteral("yolov8s.pt")); lr->setText(QStringLiteral("0.012"));
        QTRY_VERIFY_WITH_TIMEOUT(status->text().contains(QStringLiteral("已保存")), 3000);
        QMetaObject::invokeMethod(&page, "advancedRequested", Qt::DirectConnection); lr->setText(QStringLiteral("0.2")); save->click();
        QJsonObject draft; QVERIFY(settings.readTrainingDraft(query.projectIdentity(), &draft, &error));
        QCOMPARE(draft.value(QStringLiteral("controls")).toObject().value(QStringLiteral("YoloTrainArg_lr0")).toString(), QStringLiteral("0.012"));
        return;
    }
    QVERIFY2(workspace.open(root, &error), qPrintable(error)); controller.setProjectContext(true, root);
    QCOMPARE(epoch->text(), QStringLiteral("37")); QCOMPARE(model->currentText(), QStringLiteral("yolov8s.pt")); QCOMPARE(lr->text(), QStringLiteral("0.012"));
    QVERIFY(page.findChild<QLabel*>(QStringLiteral("TrainingDatasetNote"))->text().contains(QStringLiteral("已提交快照")));
    QVERIFY(page.findChild<QLabel*>(QStringLiteral("TrainingSampleNote"))->text().contains(QStringLiteral("images/a.png")));
    QCOMPARE(runtime.state(), TaskRuntimeController::State::Idle);
    const QString id = query.projectIdentity(); QJsonObject valid; QVERIFY(settings.readTrainingDraft(id, &valid, &error));
    {
        TrainingPageController delayed(&runtime); delayed.setQueryService(&query); delayed.setProjectContext(true, root);
        TrainingWorkspacePage delayedPage; delayed.attach(&delayedPage);
        QCOMPARE(delayedPage.findChild<QLineEdit*>(QStringLiteral("TrainingEpochs"))->text(), QStringLiteral("37"));
        QVERIFY(delayedPage.findChild<QLabel*>(QStringLiteral("TrainingSampleNote"))->text().contains(QStringLiteral("images/a.png")));
    }
    // 不同项目隔离；返回原项目恢复其草稿。
    controller.setProjectContext(false, {}); workspace.close();
    const QString other = QDir(base).filePath(QStringLiteral("other")); QVERIFY(workspace.createProject(other, &error)); controller.setProjectContext(true, other); QCOMPARE(epoch->text(), QStringLiteral("20"));
    controller.setProjectContext(false, {}); workspace.close(); QVERIFY(workspace.open(root, &error)); controller.setProjectContext(true, root); QCOMPARE(epoch->text(), QStringLiteral("37"));
    controller.setProjectContext(false, {});
    QJsonObject missingSample = valid; missingSample.insert(QStringLiteral("sample"), QStringLiteral("images/deleted.png")); QVERIFY(settings.saveTrainingDraft(id, missingSample));
    controller.setProjectContext(true, root); QVERIFY(status->text().contains(QStringLiteral("样本已失效")));
    controller.setProjectContext(false, {});
    QJsonObject missingSnapshot = valid; missingSnapshot.insert(QStringLiteral("datasetSnapshotId"), aitrain::SnapshotId::create().toString()); QVERIFY(settings.saveTrainingDraft(id, missingSnapshot));
    controller.setProjectContext(true, root); QCOMPARE(epoch->text(), QStringLiteral("37")); QVERIFY(status->text().contains(QStringLiteral("数据版本已失效")));
    QVERIFY(page.findChild<QLabel*>(QStringLiteral("TrainingDatasetNote"))->text().contains(QStringLiteral("尚未选择")));
    controller.setProjectContext(false, {});
    QSettings raw; QJsonObject invalid = valid; invalid.insert(QStringLiteral("version"), 999); raw.setValue(QStringLiteral("trainingDrafts/") + id, QJsonDocument(invalid).toJson()); raw.sync();
    controller.setProjectContext(true, root); QCOMPARE(epoch->text(), QStringLiteral("20")); QVERIFY(status->text().contains(QStringLiteral("格式无效")));
    // 显式丢弃后不会因退出再次写入旧草稿。
    epoch->setText(QStringLiteral("81")); save->click();
    page.resize(1024, 700); page.show(); QTest::qWait(10);
    QTimer::singleShot(100, []() { for (auto* widget : QApplication::topLevelWidgets()) if (widget->objectName() == QStringLiteral("TrainingDiscardConfirmation")) widget->findChild<QPushButton*>(QStringLiteral("TrainingConfirmDiscard"))->click(); });
    page.findChild<QPushButton*>(QStringLiteral("TrainingDiscardDraft"))->click();
    QCOMPARE(epoch->text(), QStringLiteral("20")); QJsonObject discarded; QVERIFY(!settings.readTrainingDraft(id, &discarded, &error));
    controller.setProjectContext(false, {});
    QJsonObject badBackend = valid; auto badControls = badBackend.value(QStringLiteral("controls")).toObject();
    badControls.insert(QStringLiteral("TrainingBackend"), QStringLiteral("removed_backend")); badBackend.insert(QStringLiteral("controls"), badControls);
    QVERIFY(settings.saveTrainingDraft(id, badBackend)); controller.setProjectContext(true, root);
    QVERIFY(status->text().contains(QStringLiteral("当前合同不匹配")));
    controller.setProjectContext(false, {}); QVERIFY(settings.saveTrainingDraft(id, valid)); workspace.close();
    QVERIFY2(workspace.rebuildProject(root, &error), qPrintable(error));
    QVERIFY(query.projectIdentity() != id); controller.setProjectContext(true, root);
    QCOMPARE(epoch->text(), QStringLiteral("20"));
}

void EnvironmentDeliveryEvidenceUiTests::englishAndDarkWorkbenchAreComplete()
{
    QSettings preferences;
    struct RestorePreferences {
        QVariant theme = QSettings().value(QStringLiteral("settings/theme"));
        QString language = aitrain_app::configuredLanguageCode();
        ~RestorePreferences() {
            QSettings settings;
            if (theme.isValid()) settings.setValue(QStringLiteral("settings/theme"), theme); else settings.remove(QStringLiteral("settings/theme"));
            aitrain_app::storeLanguageCode(language); AppStyle::apply(*qApp, QStringLiteral("light"));
        }
    } restore;
    aitrain_app::storeLanguageCode(QStringLiteral("en_US"));
    QTranslator english; QVERIFY(aitrain_app::loadTranslator(*qApp, &english, QStringLiteral("en_US")));
    const QRegularExpression chinese(QStringLiteral("[\\x{4e00}-\\x{9fff}]"));
    QStringList untranslated;
    for (const QString& theme : {QStringLiteral("light"), QStringLiteral("dark")}) {
        // 通过产品设置入口切换，验证持久化及即时生效。
        SettingsWorkspacePage settingsPage({}, {}); SettingsPageController settingsController(QStringLiteral(".")); settingsController.attach(&settingsPage);
        auto* themeControl = settingsPage.findChild<QComboBox*>(QStringLiteral("SettingsTheme"));
        const int selected = themeControl->findData(theme);
        themeControl->setCurrentIndex(1 - selected);
        themeControl->setCurrentIndex(selected);
        QCOMPARE(AppStyle::configuredTheme(), theme);
        QCOMPARE(qApp->palette().color(QPalette::Window).lightness() < 128, theme == QStringLiteral("dark"));
        AppStyle::apply(*qApp);
        QCOMPARE(AppStyle::configuredTheme(), theme);
        QCOMPARE(qApp->palette().color(QPalette::Window).lightness() < 128, theme == QStringLiteral("dark"));
        MainWindow window(QStringLiteral("test-license"), QStringLiteral("2099-12-31")); window.setAttribute(Qt::WA_ShowWithoutActivating, true); window.show();
        const QList<QSize> sizes = qEnvironmentVariableIntValue("AITRAIN_TEST_MINIMUM_WINDOW") ? QList<QSize>{QSize(1024, 700)} : QList<QSize>{QSize(1280, 820), QSize(1366, 768), QSize(1024, 700)};
        for (const auto& size : sizes) { window.resize(size);
            for (int index = 0; index < MainWindow::PageCount; ++index) {
                QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection, Q_ARG(int, index), Q_ARG(QString, QStringLiteral("Workspace")));
                QWidget* page = window.findChild<QStackedWidget*>(QStringLiteral("WorkspaceStack"))->currentWidget();
                auto* host = dynamic_cast<aitrain_app::WorkspaceViewHost*>(page);
                for (int mode = 0; mode < (host ? host->views->count() : 1); ++mode) {
                    if (host) host->setMode(mode); QCoreApplication::processEvents(); QCOMPARE(window.size(), size);
                    const auto check = [&](const QString& text) { if (text != QStringLiteral("中") && text != QStringLiteral("中文") && chinese.match(text).hasMatch()) untranslated.append(text); };
                    for (auto* label : page->findChildren<QLabel*>()) if (label->isVisibleTo(page)) check(label->text());
                    for (auto* button : page->findChildren<QAbstractButton*>()) if (button->isVisibleTo(page)) {
                        check(button->text()); const QRect bounds(button->mapTo(&window, QPoint()), button->size());
                        QVERIFY2(window.rect().contains(bounds), qPrintable(QStringLiteral("English button outside window: %1 / %2 / %3").arg(index).arg(mode).arg(button->text())));
                    }
                    for (auto* combo : page->findChildren<QComboBox*>()) if (combo->isVisibleTo(page)) for (int item=0;item<combo->count();++item) check(combo->itemText(item));
                    for (auto* list : page->findChildren<QListWidget*>()) if (list->isVisibleTo(page)) {
                        QCOMPARE(list->horizontalScrollBar()->maximum(), 0);
                        for (int item = 0; item < list->count(); ++item) check(list->item(item)->text());
                    }
                    const QString output = qEnvironmentVariable("AITRAIN_CAPTURE_UI_DIR");
                    if (!output.isEmpty()) { QDir().mkpath(output); QVERIFY(window.grab().save(QDir(output).filePath(QStringLiteral("en-%1-%2x%3-page%4-mode%5.png").arg(theme).arg(size.width()).arg(size.height()).arg(index).arg(mode)))); }
                }
            }
        }
        // 已创建控件随主题切换，成功/警告仍由不同文字和颜色表达。
        StatusPill pill; pill.setStatus(QStringLiteral("Ready"), StatusPill::Tone::Success); pill.show(); QCoreApplication::processEvents();
        QCOMPARE(pill.property("tone").toInt(), int(StatusPill::Tone::Success));
    }
    qApp->removeTranslator(&english); AppStyle::apply(*qApp, QStringLiteral("light"));
    untranslated.removeDuplicates(); QVERIFY2(untranslated.isEmpty(), qPrintable(untranslated.join(QLatin1Char('\n'))));
}

QTEST_MAIN(EnvironmentDeliveryEvidenceUiTests)
#include "tst_delivery_acceptance_ui.moc"
