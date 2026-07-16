#pragma once

#include <QByteArray>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcessEnvironment>
#include <QString>
#include <QStringList>

namespace worker_support {

QJsonObject checkObject(const QString& name, const QString& status, const QString& message, const QJsonObject& details = {});
bool writeJsonFile(const QString& path, const QJsonObject& object, QString* error);
QJsonObject nvidiaSmiCheck();
QString firstUsablePythonExecutable(const QJsonObject& parameters = {});
QString packagedPaddleOcrRepoPath();
void configurePackagedPythonEnvironment(QProcessEnvironment* environment);
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
QJsonObject smpEnvironmentProfile(const QString& pythonExecutable);
QJsonObject anomalibEnvironmentProfile(const QString& pythonExecutable);
QJsonObject ocrEnvironmentProfile(const QString& pythonExecutable);
QJsonObject tensorRtEnvironmentProfile(const QJsonArray& baseChecks);

} // namespace worker_support
