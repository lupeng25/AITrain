#define private public
#include "MainWindow.h"
#include "Sidebar.h"
#undef private

#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QMetaObject>
#include <QProxyStyle>
#include <QPushButton>
#include <QSettings>
#include <QStyleFactory>
#include <QStringList>
#include <QTabWidget>
#include <QTest>

#include "LanguageSupport.h"
#include "AppStyle.h"

namespace {

class NoAnimationStyle final : public QProxyStyle {
public:
    NoAnimationStyle()
        : QProxyStyle(QStyleFactory::create(QStringLiteral("Fusion")))
    {
    }

    int styleHint(StyleHint hint, const QStyleOption* option = nullptr,
        const QWidget* widget = nullptr, QStyleHintReturn* returnData = nullptr) const override
    {
        if (hint == SH_Widget_Animate) {
            return 0;
        }
        return QProxyStyle::styleHint(hint, option, widget, returnData);
    }
};

} // namespace

class EnvironmentDeliveryEvidenceUiTests : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void mainNavigationUsesNineWorkspaceEntries();
    void embeddedWorkspaceTabsExist();
    void switchingToEnvironmentShowsDeliveryEvidenceTab();
    void clickingSidebarEnvironmentSwitchesPageWithoutDeliveryEntry();
    void capabilityPanelEnglishFallbackIsComplete();

private:
    QString previousLanguage_;
    bool hadPreviousLanguage_ = false;
    MainWindow* window_ = nullptr;
};

void EnvironmentDeliveryEvidenceUiTests::initTestCase()
{
    QApplication::setStyle(new NoAnimationStyle);
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
    AppStyle::apply(*qApp);
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
            case MainWindow::DatasetPage: tabs = window_->datasetTabs_; break;
            case MainWindow::TrainingPage: tabs = window_->findChild<QTabWidget*>(QStringLiteral("TrainingDetailTabs")); break;
            case MainWindow::TaskQueuePage: tabs = window_->findChild<QTabWidget*>(QStringLiteral("TaskDetailTabs")); break;
            case MainWindow::ModelRegistryPage: tabs = window_->modelWorkspaceTabs_; break;
            case MainWindow::DeploymentPage: tabs = window_->deploymentTabs_; break;
            case MainWindow::EnvironmentPage: tabs = window_->findChild<QTabWidget*>(QStringLiteral("EnvironmentTabs")); break;
            case MainWindow::SystemSettingsPage: tabs = window_->systemSettingsTabs_; break;
            default: break;
            }
            if (tabs && captureTab >= 0 && captureTab < tabs->count()) {
                tabs->setCurrentIndex(captureTab);
            }
        }
        bool expandAdvancedOk = false;
        const int expandAdvanced = qEnvironmentVariableIntValue("AITRAIN_CAPTURE_UI_EXPAND_ADVANCED", &expandAdvancedOk);
        if (expandAdvancedOk && expandAdvanced != 0 && pageOk && capturePage == MainWindow::DeploymentPage
            && window_->deploymentTabs_) {
            if (auto* toggle = window_->deploymentTabs_->currentWidget()->findChild<QPushButton*>(QStringLiteral("AdvancedToggle"))) {
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
    const auto sidebarButtons = window.sidebar_->findChildren<QPushButton*>();
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
    QCOMPARE(modelTabs->count(), 4);
    QCOMPARE(modelTabs->tabText(0), QStringLiteral("模型版本"));
    QCOMPARE(modelTabs->tabText(1), QStringLiteral("评估报告"));
    QCOMPARE(modelTabs->tabText(2), QStringLiteral("模型对比"));
    QCOMPARE(modelTabs->tabText(3), QStringLiteral("流水线记录"));

    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::DeploymentPage), Q_ARG(QString, QStringLiteral("部署验证"))));
    auto* deploymentTabs = window.findChild<QTabWidget*>(QStringLiteral("DeploymentTabs"));
    QVERIFY(deploymentTabs != nullptr);
    QCOMPARE(deploymentTabs->count(), 2);
    QCOMPARE(deploymentTabs->tabText(0), QStringLiteral("模型导出"));
    QCOMPARE(deploymentTabs->tabText(1), QStringLiteral("推理验证"));

    QVERIFY(QMetaObject::invokeMethod(&window, "showPage", Qt::DirectConnection,
        Q_ARG(int, MainWindow::SystemSettingsPage), Q_ARG(QString, QStringLiteral("系统设置"))));
    auto* settingsTabs = window.findChild<QTabWidget*>(QStringLiteral("SystemSettingsTabs"));
    QVERIFY(settingsTabs != nullptr);
    QCOMPARE(settingsTabs->count(), 2);
    QCOMPARE(settingsTabs->tabText(0), QStringLiteral("内置能力"));
    QCOMPARE(settingsTabs->tabText(1), QStringLiteral("应用设置"));
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

    QCOMPARE(window.stack_->currentIndex(), static_cast<int>(MainWindow::EnvironmentPage));
    QCOMPARE(window.pageTitle_->text(), QStringLiteral("环境"));
    auto* tabs = window.findChild<QTabWidget*>(QStringLiteral("EnvironmentTabs"));
    QVERIFY(tabs != nullptr);
    QCOMPARE(tabs->count(), 2);
    QCOMPARE(tabs->tabText(0), QStringLiteral("运行环境"));
    QCOMPARE(tabs->tabText(1), QStringLiteral("交付证据"));

    tabs->setCurrentIndex(1);
    QCoreApplication::processEvents();

    QVERIFY(window.deliveryAcceptanceTable_ != nullptr);
    QCOMPARE(window.deliveryAcceptanceTable_->rowCount(), 7);
    QVERIFY(window.deliveryAcceptanceSummaryLabel_ != nullptr);
    QVERIFY(window.deliveryAcceptanceSummaryLabel_->text().contains(QStringLiteral("not-run 7")));
}

void EnvironmentDeliveryEvidenceUiTests::clickingSidebarEnvironmentSwitchesPageWithoutDeliveryEntry()
{
    MainWindow& window = *window_;
    window.resize(1280, 820);
    window.show();
    QVERIFY(QTest::qWaitForWindowExposed(&window));

    const auto sidebarButtons = window.sidebar_->findChildren<QPushButton*>();
    for (QPushButton* button : sidebarButtons) {
        QVERIFY(button->text() != QStringLiteral("交付验收"));
        QVERIFY(button->text() != QStringLiteral("样本复核"));
        QVERIFY(button->text() != QStringLiteral("评估报告"));
        QVERIFY(button->text() != QStringLiteral("模型导出"));
        QVERIFY(button->text() != QStringLiteral("推理验证"));
        QVERIFY(button->text() != QStringLiteral("设置"));
    }

    QPushButton* environmentButton = qobject_cast<QPushButton*>(
        window.sidebar_->buttons_.button(MainWindow::EnvironmentPage));
    QVERIFY(environmentButton != nullptr);
    QVERIFY(environmentButton->isVisible());

    environmentButton->click();
    QCoreApplication::processEvents();

    QCOMPARE(window.stack_->currentIndex(), static_cast<int>(MainWindow::EnvironmentPage));
    QCOMPARE(window.pageTitle_->text(), QStringLiteral("环境"));
    auto* tabs = window.findChild<QTabWidget*>(QStringLiteral("EnvironmentTabs"));
    QVERIFY(tabs != nullptr);
    QCOMPARE(window.deliveryAcceptanceTable_->rowCount(), 7);
    QVERIFY(window.deliveryAcceptanceSummaryLabel_->text().contains(QStringLiteral("not-run 7")));
}

void EnvironmentDeliveryEvidenceUiTests::capabilityPanelEnglishFallbackIsComplete()
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
    QStringList actual;
    for (const QString& source : sources) {
        actual.append(aitrain_app::translateText("MainWindow", source));
    }
    aitrain_app::storeLanguageCode(QStringLiteral("zh_CN"));

    QCOMPARE(actual, expected);
}

QTEST_MAIN(EnvironmentDeliveryEvidenceUiTests)
#include "tst_delivery_acceptance_ui.moc"
