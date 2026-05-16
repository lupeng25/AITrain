#pragma once

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/ProductWorkflow.h"

#include <QByteArray>
#include <QDir>
#include <QFileInfo>
#include <QFileInfoList>
#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>

namespace aitrain::workflow_detail {

QString nowIso();
QString cleanRelativePath(const QDir& root, const QString& absolutePath);
QStringList imageNameFilters();
bool writeJsonFile(const QString& path, const QJsonObject& object, QString* error);
bool writeTextFile(const QString& path, const QString& content, QString* error);
bool readJsonFile(const QString& path, QJsonObject* object, QString* error);
QByteArray fileSha256(const QString& path, qint64* size, QString* error);
QFileInfoList collectFilesRecursive(const QString& rootPath, int maxFiles);
bool isImageFile(const QString& suffix);
QString firstImageFileUnder(const QString& rootPath);
QJsonObject fileDigestObject(const QString& path, QString* error);
QJsonObject pathArtifact(const QString& kind, const QString& path, const QString& message = QString());
QJsonObject readJsonObjectIfExists(const QString& path);
QJsonObject evaluationDeliverySummary(const QString& reportPath);
QJsonObject benchmarkDeliverySummary(const QString& reportPath);
QJsonArray deliveryLimitations(const QJsonObject& context, const QJsonObject& evaluationSummary, const QJsonObject& benchmarkSummary);
DatasetValidationResult validateByFormat(const QString& datasetPath, const QString& format, const QJsonObject& options);
QString csvEscape(QString value);
QString htmlEscape(QString value);
WorkflowResult resultFromReport(const QString& reportPath, const QJsonObject& payload);
WorkflowResult failedResult(const QString& error);
WorkflowResult canceledResult();

} // namespace aitrain::workflow_detail
