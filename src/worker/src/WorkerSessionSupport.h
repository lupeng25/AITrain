#pragma once

#include "aitrain/core/TaskModels.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>

namespace worker_support {

QJsonObject checkObject(const QString& name, const QString& status, const QString& message, const QJsonObject& details = {});
bool writeJsonFile(const QString& path, const QJsonObject& object, QString* error);
QString defaultTaskOutputPath(const QString& basePath, const QString& taskId);
QJsonObject nvidiaSmiCheck();
QString firstUsablePythonExecutable(const QJsonObject& parameters = {});
QString pythonTrainerScriptFileForBackend(const QString& backend);
QString pythonTrainerScriptPath(const QJsonObject& parameters, const QString& backend);
QString requestedTrainingBackend(const aitrain::TrainingRequest& request);
bool isPythonTrainingBackendId(const QString& backend, const QJsonObject& parameters);
QJsonObject runPythonCommandCheck(
    const QString& name,
    const QString& executable,
    const QStringList& arguments,
    int timeoutMs,
    const QString& missingMessage);
QJsonObject pythonModuleCheck(const QString& executable, const QString& displayName, const QString& moduleName, const QString& missingMessage);
QJsonObject profileCheck(const QString& name, const QString& status, const QString& message, const QJsonObject& details = {});
QJsonObject makeProfile(const QString& id, const QString& title, const QJsonArray& checks, const QJsonArray& repairHints);
QJsonObject runModuleProbe(const QString& pythonExecutable, const QString& checkName, const QString& moduleName, const QString& hint);
QJsonObject yoloEnvironmentProfile(const QString& pythonExecutable);
QJsonObject ocrEnvironmentProfile(const QString& pythonExecutable);
QJsonObject tensorRtEnvironmentProfile(const QJsonArray& baseChecks);

} // namespace worker_support
