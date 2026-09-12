#pragma once

#include <QString>
#include <QJsonObject>
#include <QSettings>

namespace aitrain_app {

class ApplicationSettingsService
{
public:
    ApplicationSettingsService();

    bool readTrainingDraft(const QString& projectId, QJsonObject* draft, QString* error) const;
    bool saveTrainingDraft(const QString& projectId, const QJsonObject& draft);
    void removeTrainingDraft(const QString& projectId);

    QString languageCode(const QString& fallback) const;
    void setLanguageCode(const QString& languageCode);

    QString defaultProjectPath(const QString& fallback) const;
    void setDefaultProjectPath(const QString& path);

private:
    QSettings settings_;
};

} // namespace aitrain_app
