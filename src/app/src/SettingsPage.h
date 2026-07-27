#pragma once

#include <QVector>
#include <QWidget>

class QLabel;
class QLineEdit;
class QTabWidget;
class QTableWidget;
class QToolButton;

struct SettingsCapabilityRow final
{
    QString id;
    QString displayName;
    QString taskTypes;
    QString datasetFormats;
    QString backendIds;
};

struct SettingsCapabilitySummary final
{
    int capabilityCount = 0;
    int datasetFormatCount = 0;
    int exportFormatCount = 0;
    int gpuCapabilityCount = 0;
    QString datasetFormats;
    QString exportFormats;
};

class SettingsWorkspacePage final : public QWidget
{
    Q_OBJECT

public:
    explicit SettingsWorkspacePage(const QString& licenseOwner,
        const QString& licenseExpiry, QWidget* parent = nullptr);

    QString defaultProjectPathText() const;
    void setDefaultProjectPathText(const QString& path);
    void setDefaultProjectPathStatus(const QString& status);
    void setLanguageCode(const QString& languageCode);
    void setCapabilities(const QVector<SettingsCapabilityRow>& rows,
        const SettingsCapabilitySummary& summary);
    void showTab(int tabIndex);

signals:
    void refreshCapabilitiesRequested();
    void languageRequested(const QString& languageCode);
    void browseDefaultProjectPathRequested();
    void saveDefaultProjectPathRequested();
    void resetDefaultProjectPathRequested();
    void openProjectRequested();
    void openEnvironmentRequested();
    void runEnvironmentRequested();

private:
    QWidget* buildCapabilitiesPanel();
    QWidget* buildApplicationSettingsPanel(
        const QString& licenseOwner, const QString& licenseExpiry);

    QTabWidget* tabs_ = nullptr;
    QToolButton* zhLanguageButton_ = nullptr;
    QToolButton* enLanguageButton_ = nullptr;
    QLineEdit* defaultProjectPathEdit_ = nullptr;
    QLabel* defaultProjectPathStatusLabel_ = nullptr;
    QLabel* capabilityStatusLabel_ = nullptr;
    QLabel* capabilitySourceLabel_ = nullptr;
    QLabel* capabilityCountLabel_ = nullptr;
    QLabel* datasetFormatCountLabel_ = nullptr;
    QLabel* exportFormatCountLabel_ = nullptr;
    QLabel* gpuCapabilityCountLabel_ = nullptr;
    QTableWidget* capabilityTable_ = nullptr;
};
