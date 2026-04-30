#pragma once

#include <QJsonObject>
#include <QString>
#include <QStringList>
#include <QVector>

namespace aitrain {

struct PackagingLayout {
    QString rootPath;
    QString appExecutablePath;
    QString workerExecutablePath;
    QString pluginModelsDirectory;
    QString runtimesDirectory;
    QString onnxRuntimeDirectory;
    QString tensorRtRuntimeDirectory;
    QString examplesDirectory;
    QString docsDirectory;

    QJsonObject toJson() const;
};

struct RuntimeDependencyCheck {
    QString name;
    QStringList libraryNames;
    QString status;
    QString message;
    QString resolvedPath;
    QStringList searchPaths;
    QJsonObject details;

    QJsonObject toJson() const;
};

PackagingLayout packagingLayoutForRoot(const QString& rootPath);

QStringList runtimeSearchPaths(const QString& applicationDir = QString());

RuntimeDependencyCheck checkRuntimeDependency(
    const QString& name,
    const QStringList& libraryNames,
    const QString& missingMessage,
    const QString& applicationDir = QString());

QVector<RuntimeDependencyCheck> defaultRuntimeDependencyChecks(const QString& applicationDir = QString());

} // namespace aitrain
