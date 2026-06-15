#define private public
#include "MainWindow.h"
#include "Sidebar.h"
#undef private

#include <QMetaObject>
#include <QPushButton>
#include <QTabWidget>
#include <QTest>

class EnvironmentDeliveryEvidenceUiTests : public QObject {
    Q_OBJECT

private slots:
    void switchingToEnvironmentShowsDeliveryEvidenceTab();
    void clickingSidebarEnvironmentSwitchesPageWithoutDeliveryEntry();
};

void EnvironmentDeliveryEvidenceUiTests::switchingToEnvironmentShowsDeliveryEvidenceTab()
{
    MainWindow window(QStringLiteral("test-license"), QStringLiteral("2099-12-31"));

    const bool invoked = QMetaObject::invokeMethod(
        &window,
        "showPage",
        Qt::DirectConnection,
        Q_ARG(int, MainWindow::EnvironmentPage),
        Q_ARG(QString, QStringLiteral("环境")));
    QVERIFY(invoked);

    QCOMPARE(window.stack_->currentIndex(), static_cast<int>(MainWindow::EnvironmentPage));
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
    MainWindow window(QStringLiteral("test-license"), QStringLiteral("2099-12-31"));
    window.resize(1280, 820);
    window.show();
    QVERIFY(QTest::qWaitForWindowExposed(&window));

    const auto sidebarButtons = window.sidebar_->findChildren<QPushButton*>();
    for (QPushButton* button : sidebarButtons) {
        QVERIFY(button->text() != QStringLiteral("交付验收"));
    }

    QPushButton* environmentButton = qobject_cast<QPushButton*>(
        window.sidebar_->buttons_.button(MainWindow::EnvironmentPage));
    QVERIFY(environmentButton != nullptr);
    QVERIFY(environmentButton->isVisible());

    environmentButton->click();
    QCoreApplication::processEvents();

    QCOMPARE(window.stack_->currentIndex(), static_cast<int>(MainWindow::EnvironmentPage));
    auto* tabs = window.findChild<QTabWidget*>(QStringLiteral("EnvironmentTabs"));
    QVERIFY(tabs != nullptr);
    QCOMPARE(window.deliveryAcceptanceTable_->rowCount(), 7);
    QVERIFY(window.deliveryAcceptanceSummaryLabel_->text().contains(QStringLiteral("not-run 7")));
}

QTEST_MAIN(EnvironmentDeliveryEvidenceUiTests)
#include "tst_delivery_acceptance_ui.moc"
