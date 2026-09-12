#include "WorkbenchTranslation.h"
#include "ApplicationSettingsService.h"
#include <QJsonDocument>
#include <QUuid>

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

bool aitrain_app::ApplicationSettingsService::readTrainingDraft(const QString& projectId, QJsonObject* draft, QString* error) const
{
    if (error) error->clear();
    if (!draft || QUuid(projectId).isNull()) return false;
    *draft = {};
    const QByteArray bytes = settings_.value(QStringLiteral("trainingDrafts/") + projectId).toByteArray();
    if (bytes.isEmpty()) return false;
    const auto document = bytes.size() <= 256 * 1024 ? QJsonDocument::fromJson(bytes) : QJsonDocument();
    const auto object = document.object();
    if (!document.isObject() || object.value(QStringLiteral("version")).toInt() != 1
        || object.value(QStringLiteral("projectId")).toString() != projectId
        || !object.value(QStringLiteral("controls")).isObject()) {
        if (error) *error = aitrain_app::workbenchText(QStringLiteral("草稿格式无效，已保留默认参数。"));
        return false;
    }
    *draft = object; return true;
}

bool aitrain_app::ApplicationSettingsService::saveTrainingDraft(const QString& projectId, const QJsonObject& draft)
{
    if (QUuid(projectId).isNull()) return false;
    QJsonObject object = draft;
    object.insert(QStringLiteral("version"), 1); object.insert(QStringLiteral("projectId"), projectId);
    const auto bytes = QJsonDocument(object).toJson(QJsonDocument::Compact);
    if (bytes.size() > 256 * 1024) return false;
    settings_.setValue(QStringLiteral("trainingDrafts/") + projectId, bytes); settings_.sync();
    return settings_.status() == QSettings::NoError;
}

void aitrain_app::ApplicationSettingsService::removeTrainingDraft(const QString& projectId)
{
    if (QUuid(projectId).isNull()) return;
    settings_.remove(QStringLiteral("trainingDrafts/") + projectId); settings_.sync();
}
