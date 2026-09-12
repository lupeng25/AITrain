#pragma once

#include <QJsonObject>
#include <QWidget>

class QLabel;
class QTableWidget;

class EnvironmentWorkspacePage final : public QWidget
{
    Q_OBJECT

public:
    explicit EnvironmentWorkspacePage(QWidget* deliveryEvidencePage,
        QWidget* parent = nullptr);

    void renderReport(const QJsonObject& report);
    void setChecking();

signals:
    void runRequested();
    void diagnosticsRequested();

private:
    void updateSummary();

    QLabel* statusLabel_ = nullptr;
    QLabel* okLabel_ = nullptr;
    QLabel* warningLabel_ = nullptr;
    QLabel* missingLabel_ = nullptr;
    QLabel* uncheckedLabel_ = nullptr;
    QTableWidget* table_ = nullptr;
};
