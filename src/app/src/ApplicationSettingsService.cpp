#include "ApplicationSettingsService.h"

namespace aitrain_app {

ApplicationSettingsService::ApplicationSettingsService()
{
}

QString ApplicationSettingsService::languageCode(const QString& fallback) const
{
    return settings_.value(QStringLiteral("settings/language"), fallback).toString();
}

void ApplicationSettingsService::setLanguageCode(const QString& languageCode)
{
    settings_.setValue(QStringLiteral("settings/language"), languageCode);
    settings_.sync();
}

QString ApplicationSettingsService::defaultProjectPath(const QString& fallback) const
{
    return settings_.value(QStringLiteral("settings/defaultProjectPath"), fallback).toString();
}

void ApplicationSettingsService::setDefaultProjectPath(const QString& path)
{
    settings_.setValue(QStringLiteral("settings/defaultProjectPath"), path);
    settings_.sync();
}

} // namespace aitrain_app
