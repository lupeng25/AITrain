#include "MainWindow.h"
#include "Sidebar.h"
#include "WorkerClient.h"
#include "WorkspaceRouter.h"
#include "aitrain/core/WorkerProtocol.h"

#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QFrame>
#include <QLabel>
#include <QLineEdit>
#include <QMetaObject>
#include <QPushButton>
#include <QSettings>
#include <QSignalSpy>
#include <QStackedWidget>
#include <QStringList>
#include <QTabWidget>
#include <QTableWidget>
#include <QTest>
#include <QTemporaryDir>
#include <QTranslator>
#include <QUuid>

#include "LanguageSupport.h"

class EnvironmentDeliveryEvidenceUiTests : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void mainNavigationUsesNineWorkspaceEntries();
    void embeddedWorkspaceTabsExist();
    void switchingToEnvironmentShowsDeliveryEvidenceTab();
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

private:
    QString previousLanguage_;
    bool hadPreviousLanguage_ = false;
    MainWindow* window_ = nullptr;
    QTranslator translator_;
};

void EnvironmentDeliveryEvidenceUiTests::initTestCase()
{
    QApplication::setStyle(QStringLiteral("Fusion"));
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

void EnvironmentDeliveryEvidenceUiTests::mainNavigationUsesNineWorkspaceEntries()
{
    MainWindow& window = *window_;

    QStringList labels;
    auto* sidebar = window.findChild<Sidebar*>(QStringLiteral("WorkspaceSidebar"));
    QVERIFY(sidebar != nullptr);
    const auto sidebarButtons = sidebar->findChildren<QPushButton*>();
    for (QPushButton* button : sidebarButtons) {
        if (button->objectName() == QStringLiteral("SidebarButton")) {
            const QString fullText = button->property("fullText").toString();
            labels << (fullText.isEmpty() ? button->text() : fullText);
        }
    }

    QCOMPARE(labels.size(), 9);
    QCOMPARE(labels, QStringList()
        << QStringLiteral("总览")
        << QStringLiteral("项目")
        << QStringLiteral("数据集")
        << QStringLiteral("训练实验")
        << QStringLiteral("任务与产物")
        << QStringLiteral("模型库")
        << QStringLiteral("部署验证")
        << QStringLiteral("环境")
        << QStringLiteral("系统设置"));

    const QStringList removedEntries = {
        QStringLiteral("样本复核"),
        QStringLiteral("评估报告"),
        QStringLiteral("模型导出"),
        QStringLiteral("推理验证"),
        QStringLiteral("设置"),
        QStringLiteral("交付验收")
    };
    for (const QString& entry : removedEntries) {
        QVERIFY(!labels.contains(entry));
    }
}

void EnvironmentDeliveryEvidenceUiTests::embeddedWorkspaceTabsExist()
{
    MainWindow& window = *window_;

    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::DatasetPage), Q_ARG(QString, QStringLiteral("数据集"))));
    auto* datasetTabs = window.findChild<QTabWidget*>(QStringLiteral("DatasetTabs"));
    QVERIFY(datasetTabs != nullptr);
    QCOMPARE(datasetTabs->count(), 2);
    QCOMPARE(datasetTabs->tabText(0), QStringLiteral("数据集准备"));
    QCOMPARE(datasetTabs->tabText(1), QStringLiteral("质量与复核"));

    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::ModelRegistryPage), Q_ARG(QString, QStringLiteral("模型库"))));
    auto* modelTabs = window.findChild<QTabWidget*>(QStringLiteral("ModelWorkspaceTabs"));
    QVERIFY(modelTabs != nullptr);
    QCOMPARE(modelTabs->count(), 1);
    QCOMPARE(modelTabs->tabText(0), QStringLiteral(" 模型包"));

    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::DeploymentPage), Q_ARG(QString, QStringLiteral("部署验证"))));
    auto* deploymentTabs = window.findChild<QTabWidget*>(QStringLiteral("DeploymentTabs"));
    QVERIFY(deploymentTabs != nullptr);
    QCOMPARE(deploymentTabs->count(), 2);
        QCOMPARE(deploymentTabs->tabText(0), QStringLiteral("部署验证"));
    QCOMPARE(deploymentTabs->tabText(1), QStringLiteral("推理验证"));

    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::SystemSettingsPage), Q_ARG(QString, QStringLiteral("系统设置"))));
    auto* settingsTabs = window.findChild<QTabWidget*>(QStringLiteral("SystemSettingsTabs"));
    QVERIFY(settingsTabs != nullptr);
    QCOMPARE(settingsTabs->count(), 2);
    QCOMPARE(settingsTabs->tabText(0), QStringLiteral("内置能力"));
    QCOMPARE(settingsTabs->tabText(1), QStringLiteral("应用设置"));
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
    QVERIFY(window.findChild<QTableWidget*>(QStringLiteral("TaskQueueTable")) != nullptr);
    QVERIFY(window.findChild<QTableWidget*>(QStringLiteral("TaskArtifactTable")) != nullptr);
    QVERIFY(window.findChild<QTableWidget*>(QStringLiteral("TaskMetricTable")) != nullptr);
    QVERIFY(window.findChild<QTableWidget*>(QStringLiteral("TaskWorkflowTable")) != nullptr);
    auto* cancelButton = window.findChild<QPushButton*>(QStringLiteral("TaskCancelButton"));
    QVERIFY(cancelButton != nullptr);
    QVERIFY(!cancelButton->isEnabled());
    for (QPushButton* button : window.findChildren<QPushButton*>()) {
        QVERIFY2(button->text() != QStringLiteral("复现实验"),
            "任务页不得重新引入未接线的复现实验入口");
    }
    auto* detailTabs = window.findChild<QTabWidget*>(QStringLiteral("TaskDetailTabs"));
    QVERIFY(detailTabs != nullptr);
    QCOMPARE(detailTabs->tabText(2), QStringLiteral("工作流"));
}

void EnvironmentDeliveryEvidenceUiTests::annotationSessionUiUsesArtifactBoundary()
{
    MainWindow& window = *window_;
    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::DatasetPage), Q_ARG(QString, QStringLiteral("数据集"))));
    QVERIFY(window.findChild<QPushButton*>(QStringLiteral("CreateAnnotationSessionButton")) != nullptr);
    QVERIFY(window.findChild<QPushButton*>(QStringLiteral("SyncAnnotationSessionButton")) != nullptr);

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
        Q_ARG(int, MainWindow::EnvironmentPage), Q_ARG(QString, QStringLiteral("环境"))));
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

    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::TaskQueuePage), Q_ARG(QString, QStringLiteral("任务与产物"))));
    QVERIFY(window.findChild<QFrame*>(QStringLiteral("ArtifactActionGrid")) == nullptr);
    const auto buttons = window.findChildren<QPushButton*>();
    for (const QPushButton* button : buttons) {
        QVERIFY(button->text() != QStringLiteral("交付报告"));
    }
}

void EnvironmentDeliveryEvidenceUiTests::switchingToEnvironmentShowsDeliveryEvidenceTab()
{
    MainWindow& window = *window_;

    const bool invoked = QMetaObject::invokeMethod(
        &window,
        "showPage",
        Qt::DirectConnection,
        Q_ARG(int, MainWindow::EnvironmentPage),
        Q_ARG(QString, QStringLiteral("环境")));
    QVERIFY(invoked);

    auto* stack = window.findChild<QStackedWidget*>(QStringLiteral("WorkspaceStack"));
    auto* pageTitle = window.findChild<QLabel*>(QStringLiteral("PageTitle"));
    QVERIFY(stack != nullptr);
    QVERIFY(pageTitle != nullptr);
    QCOMPARE(stack->currentIndex(), static_cast<int>(MainWindow::EnvironmentPage));
    auto* router = window.findChild<WorkspaceRouter*>(QStringLiteral("WorkspaceRouter"));
    QVERIFY(router != nullptr);
    QCOMPARE(router->currentPageIndex(), static_cast<int>(MainWindow::EnvironmentPage));
    QCOMPARE(router->currentTitle(), QStringLiteral("环境"));
    QCOMPARE(pageTitle->text(), QStringLiteral("环境"));
    auto* tabs = window.findChild<QTabWidget*>(QStringLiteral("EnvironmentTabs"));
    QVERIFY(tabs != nullptr);
    QCOMPARE(tabs->count(), 2);
    QCOMPARE(tabs->tabText(0), QStringLiteral("运行环境"));
    QCOMPARE(tabs->tabText(1), QStringLiteral("交付证据"));

    tabs->setCurrentIndex(1);
    QCoreApplication::processEvents();

    auto* table = window.findChild<QTableWidget*>(QStringLiteral("DeliveryAcceptanceTable"));
    auto* summary = window.findChild<QLabel*>(QStringLiteral("DeliveryAcceptanceSummary"));
    QVERIFY(table != nullptr);
    QCOMPARE(table->rowCount(), 7);
    QVERIFY(summary != nullptr);
    QVERIFY(summary->text().contains(QStringLiteral("not-run 7")));
}

void EnvironmentDeliveryEvidenceUiTests::clickingSidebarEnvironmentSwitchesPageWithoutDeliveryEntry()
{
    MainWindow& window = *window_;
    window.resize(1280, 820);
    window.show();
    QVERIFY(QTest::qWaitForWindowExposed(&window));

    auto* sidebar = window.findChild<Sidebar*>(QStringLiteral("WorkspaceSidebar"));
    QVERIFY(sidebar != nullptr);
    const auto sidebarButtons = sidebar->findChildren<QPushButton*>();
    for (QPushButton* button : sidebarButtons) {
        QVERIFY(button->text() != QStringLiteral("交付验收"));
        QVERIFY(button->text() != QStringLiteral("样本复核"));
        QVERIFY(button->text() != QStringLiteral("评估报告"));
        QVERIFY(button->text() != QStringLiteral("模型导出"));
        QVERIFY(button->text() != QStringLiteral("推理验证"));
        QVERIFY(button->text() != QStringLiteral("设置"));
    }

    QPushButton* environmentButton = nullptr;
    for (QPushButton* button : sidebarButtons) {
        if (button->property("fullText").toString() == QStringLiteral("环境")) {
            environmentButton = button;
            break;
        }
    }
    QVERIFY(environmentButton != nullptr);
    QVERIFY(environmentButton->isVisible());

    environmentButton->click();
    QCoreApplication::processEvents();

    auto* stack = window.findChild<QStackedWidget*>(QStringLiteral("WorkspaceStack"));
    auto* pageTitle = window.findChild<QLabel*>(QStringLiteral("PageTitle"));
    QVERIFY(stack != nullptr);
    QVERIFY(pageTitle != nullptr);
    QCOMPARE(stack->currentIndex(), static_cast<int>(MainWindow::EnvironmentPage));
    QCOMPARE(pageTitle->text(), QStringLiteral("环境"));
    auto* tabs = window.findChild<QTabWidget*>(QStringLiteral("EnvironmentTabs"));
    QVERIFY(tabs != nullptr);
    auto* table = window.findChild<QTableWidget*>(QStringLiteral("DeliveryAcceptanceTable"));
    auto* summary = window.findChild<QLabel*>(QStringLiteral("DeliveryAcceptanceSummary"));
    QVERIFY(table != nullptr);
    QVERIFY(summary != nullptr);
    QCOMPARE(table->rowCount(), 7);
    QVERIFY(summary->text().contains(QStringLiteral("not-run 7")));
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
    QString error;
    QElapsedTimer elapsed;
    elapsed.start();
    const aitrain::worker_protocol::TaskCommand command{
        aitrain::worker_protocol::EnvironmentCheckCommand{
            {aitrain::TaskId::create(), directory.path()}}};
    QVERIFY2(client.startTask(invalidWorker, command, &error), qPrintable(error));
    QVERIFY2(elapsed.elapsed() < 500, "Worker 启动请求不应同步等待进程创建结果。");
    QVERIFY2(finishedSpy.wait(5000), "异步启动失败必须通过 finished 信号收口。");
    QCOMPARE(finishedSpy.count(), 1);
    QCOMPARE(qvariant_cast<WorkerClient::WorkerTerminalStatus>(finishedSpy.first().at(0)),
        WorkerClient::WorkerTerminalStatus::Failed);
    QVERIFY(finishedSpy.first().at(1).toString().contains(QStringLiteral("failed to start"), Qt::CaseInsensitive));
}

void EnvironmentDeliveryEvidenceUiTests::runtimeDeliveryPagesExposeOneSixStepProductEntry()
{
    MainWindow& window = *window_;
    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::DeploymentPage), Q_ARG(QString, QStringLiteral("部署验证"))));
    auto* tabs = window.findChild<QTabWidget*>(QStringLiteral("DeploymentTabs"));
    QVERIFY(tabs != nullptr);
    int unifiedEntryCount = 0;
    for (QPushButton* button : tabs->findChildren<QPushButton*>()) {
        if (button->text() == QStringLiteral("运行完整 Runtime Delivery")) ++unifiedEntryCount;
        QVERIFY(button->text() != QStringLiteral("开始推理"));
        QVERIFY(button->text() != QStringLiteral("开始部署验证"));
    }
    QCOMPARE(unifiedEntryCount, 2);
    bool sixStepTextVisible = false;
    for (QLabel* label : tabs->findChildren<QLabel*>()) {
        if (label->text().contains(QStringLiteral("Runtime Delivery 六步"))) {
            sixStepTextVisible = true;
            break;
        }
    }
    QVERIFY(sixStepTextVisible);
}

QTEST_MAIN(EnvironmentDeliveryEvidenceUiTests)
#include "tst_delivery_acceptance_ui.moc"
