#pragma once

#include "ApplicationSettingsService.h"

#include <QObject>
#include <QString>

class SettingsWorkspacePage;
class QWidget;

class SettingsPageController final : public QObject
{
    Q_OBJECT

public:
    explicit SettingsPageController(QString defaultProjectPath,
        QObject* parent = nullptr);

    void attach(SettingsWorkspacePage* page);
    void refresh();
    void refreshCapabilities();
    QString configuredDefaultProjectPath() const;
    void showTab(int tabIndex);
    void setLanguageCode(const QString& languageCode);

signals:
    void languageChanged(const QString& languageCode);
    void defaultProjectPathChanged(const QString& path);

private:
    void browseDefaultProjectPath();
    void saveDefaultProjectPath();
    void resetDefaultProjectPath();
    QString normalizedDefaultProjectPath(const QString& path) const;

    aitrain_app::ApplicationSettingsService settings_;
    QString defaultProjectPath_;
    SettingsWorkspacePage* page_ = nullptr;
};
