#define private public
#include "MainWindow.h"
#include "Sidebar.h"
#undef private

#include <QMetaObject>
#include <QPushButton>
#include <QTest>

class DeliveryAcceptanceUiTests : public QObject {
    Q_OBJECT

private slots:
    void switchingToDeliveryAcceptanceInitializesRows();
    void clickingSidebarDeliveryAcceptanceSwitchesPage();
};

void DeliveryAcceptanceUiTests::switchingToDeliveryAcceptanceInitializesRows()
{
    MainWindow window(QStringLiteral("test-license"), QStringLiteral("2099-12-31"));
    QVERIFY(window.deliveryAcceptanceTable_ != nullptr);
    QCOMPARE(window.deliveryAcceptanceTable_->rowCount(), 0);

    const bool invoked = QMetaObject::invokeMethod(
        &window,
        "showPage",
        Qt::DirectConnection,
        Q_ARG(int, MainWindow::DeliveryAcceptancePage),
        Q_ARG(QString, QStringLiteral("交付验收")));
    QVERIFY(invoked);

    QCOMPARE(window.stack_->currentIndex(), static_cast<int>(MainWindow::DeliveryAcceptancePage));
    QVERIFY(window.deliveryAcceptanceTable_ != nullptr);
    QCOMPARE(window.deliveryAcceptanceTable_->rowCount(), 7);
    QVERIFY(window.deliveryAcceptanceSummaryLabel_ != nullptr);
    QVERIFY(window.deliveryAcceptanceSummaryLabel_->text().contains(QStringLiteral("not-run 7")));
}

void DeliveryAcceptanceUiTests::clickingSidebarDeliveryAcceptanceSwitchesPage()
{
    MainWindow window(QStringLiteral("test-license"), QStringLiteral("2099-12-31"));
    window.resize(1280, 820);
    window.show();
    QVERIFY(QTest::qWaitForWindowExposed(&window));

    QPushButton* deliveryButton = qobject_cast<QPushButton*>(
        window.sidebar_->buttons_.button(MainWindow::DeliveryAcceptancePage));
    QVERIFY(deliveryButton != nullptr);
    QVERIFY(deliveryButton->isVisible());

    deliveryButton->click();
    QCoreApplication::processEvents();

    QCOMPARE(window.stack_->currentIndex(), static_cast<int>(MainWindow::DeliveryAcceptancePage));
    QCOMPARE(window.deliveryAcceptanceTable_->rowCount(), 7);
    QVERIFY(window.deliveryAcceptanceSummaryLabel_->text().contains(QStringLiteral("not-run 7")));
}

QTEST_MAIN(DeliveryAcceptanceUiTests)
#include "tst_delivery_acceptance_ui.moc"
