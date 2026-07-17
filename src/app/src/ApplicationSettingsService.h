#pragma once

#include <QString>
#include <QSettings>

namespace aitrain_app {

class ApplicationSettingsService
{
public:
    ApplicationSettingsService();

    QString languageCode(const QString& fallback) const;
    void setLanguageCode(const QString& languageCode);

    QString defaultProjectPath(const QString& fallback) const;
    void setDefaultProjectPath(const QString& path);

private:
    QSettings settings_;
};

} // namespace aitrain_app
